#!/usr/bin/env python3

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
import verdict
from verdict import Pipeline, Layer
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.schema import Schema, Field
from verdict.common.judge import JudgeUnit
from verdict.transform import MapUnit, MaxPoolUnit
from collections import Counter
from verdict.util import ratelimit
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field

load_dotenv()

@dataclass
class Config:
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None

    gpt_model_4o: str = "gpt-4o"
    gpt_model_o3_mini: str = "o3-mini"
    claude_model: str = "claude-3-5-sonnet-20241022"
    gemini_model: str = "gemini-1.5-pro"
    grok_model: str = "grok-2-1212"

    ground_truth_path: Path = Path("data/ground_truth.pkl")
    redacted_papers_path: Path = Path("data/redacted_papers.pkl")
    output_dir: Path = Path("results")
    prompt_dir: Path = Path("prompts")
    
    def __post_init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.grok_api_key = os.getenv('GROK_API_KEY')

def setup_llm_providers(config: Config) -> Dict[str, Any]:
    providers = {}

    if config.openai_api_key:
        providers["openai"] = OpenAI(api_key=config.openai_api_key)
    if config.anthropic_api_key:
        providers["anthropic"] = Anthropic(api_key=config.anthropic_api_key)
    if config.gemini_api_key:
        providers["gemini"] = genai.configure(api_key=config.gemini_api_key)
    if config.grok_api_key:
        providers["grok"] = OpenAI(api_key=config.grok_api_key, base_url="https://grok.openai.com/v1")

    if len(providers) != 4:
        configured_providers = list(providers.keys())
        raise ValueError(f"Missing API keys. Configured providers: {', '.join(configured_providers)}. Need all 4 providers.")
    
    return providers

def load_prompt(filename: str):
    with open(f"prompts/{filename}.txt", "r", encoding="utf-8") as f:
        return f.read()

def load_data(config: Config):
    with open(config.ground_truth_path, "rb") as f:
        ground_truth = pickle.load(f)
    with open(config.redacted_papers_path, "rb") as f:
        redacted_papers = pickle.load(f)
    return ground_truth, redacted_papers    

# Pydantic models for structured output (same as notebook)
class Section(BaseModel):
    name: str
    content: str

class Outline(BaseModel):
    proposed_method: str = Field(..., description="""Using the given information, first provide inspiration behind a new proposed method to address the main research problem.
                                                    You should also motivate why the proposed method would work better than existing works. Then, explain how the proposed
                                                    approach works, and describe all the essential steps. Do NOT repeat proposed methods that are already in the 'attempted_methods.'""")
    experimental_plan: str = Field(..., description="""Break down EVERY single step in 'proposed_method'. Every step MUST be executable.
                                                    Cover ALL essential details such as the datasets, models, metrics to be used, etc.""")  

class Contributions(BaseModel):
    contributions: List[Section] = Field(..., description="The contributions section will include ALL of the following sections: Methods, Experiments.")

# API call functions for different providers
def api_call_openai(client: OpenAI, prompt: str, inputs: List[str], model: str, temperature: float = 0.0):
    messages = [{"role": "system", "content": prompt}]
    for user_input in inputs:
        messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

def api_call_anthropic(client: Anthropic, prompt: str, inputs: List[str], model: str, temperature: float = 0.0):
    messages = [{"role": "user", "content": prompt + "\n\n" + "\n\n".join(inputs)}]
    
    response = client.messages.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.content[0].text

def api_call_gemini(client, prompt: str, inputs: List[str], model: str, temperature: float = 0.0):
    full_prompt = prompt + "\n\n" + "\n\n".join(inputs)
    
    response = client.generate_content(
        model=model,
        contents=full_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return response.text

# Two-stage prediction functions (same structure as notebook)
def outline(provider: str, client: Any, prompt: str, redacted_paper: str, attempted_methods: str, model: str):
    if provider == "openai":
        return api_call_openai(client, prompt, [redacted_paper, attempted_methods], model, 0.8)
    elif provider == "anthropic":
        return api_call_anthropic(client, prompt, [redacted_paper, attempted_methods], model, 0.8)
    elif provider == "gemini":
        return api_call_gemini(client, prompt, [redacted_paper, attempted_methods], model, 0.8)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def write_contributions(provider: str, client: Any, prompt: str, outline_paper: str, model: str):
    if provider == "openai":
        return api_call_openai(client, prompt, [outline_paper], model, 0.8)
    elif provider == "anthropic":
        return api_call_anthropic(client, prompt, [outline_paper], model, 0.8)
    elif provider == "gemini":
        return api_call_gemini(client, prompt, [outline_paper], model, 0.8)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def run_predictions(config: Config, providers: Dict[str, Any], redacted_papers: List[Tuple[str, str]], 
                   outline_prompt: str, writing_prompt: str, tries: int = 1):
    """
    Run the two-stage prediction pipeline for all models except Grok
    """
    models = {
        "openai": [config.gpt_model_4o, config.gpt_model_o3_mini],
        "anthropic": [config.claude_model],
        "gemini": [config.gemini_model]
    }
    
    all_predictions = {}
    
    for provider, model_list in models.items():
        if provider not in providers:
            continue
            
        client = providers[provider]
        all_predictions[provider] = {}
        
        for model in model_list:
            print(f"\nProcessing {provider} - {model}")
            predictions = []
            base_dir = config.output_dir / f"{provider}_{model.replace('-', '_')}"
            base_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (custom_id, redacted_paper) in enumerate(tqdm(redacted_papers, desc=f"{provider}-{model}")):
                paper_dir = base_dir / custom_id
                paper_dir.mkdir(exist_ok=True)
                
                attempted_methods = []
                
                for i in range(tries):
                    try:
                        # Stage 1: Generate outline
                        outline_paper = outline(provider, client, outline_prompt, redacted_paper, 
                                              json.dumps({"attempted_methods": attempted_methods}), model)
                        
                        with open(paper_dir / f'outline_{i+1}.txt', "w") as f:
                            f.write(outline_paper)
                        
                        # Track attempted methods to avoid repetition
                        try:
                            outline_data = json.loads(outline_paper)
                            attempted_methods.append(outline_data.get('proposed_method', ''))
                        except json.JSONDecodeError:
                            pass
                        
                        # Stage 2: Generate contributions
                        contributions = write_contributions(provider, client, writing_prompt, outline_paper, model)
                        
                        predictions.append(contributions)
                        with open(paper_dir / f'contributions_{i+1}.txt', "w") as f:
                            f.write(contributions)
                            
                    except Exception as e:
                        print(f"Error processing paper {idx} with {provider}-{model}: {e}")
                        continue
            
            # Save predictions for this model
            with open(base_dir / 'predictions.pkl', 'wb') as f:
                pickle.dump(predictions, f)
            
            all_predictions[provider][model] = predictions
    
    return all_predictions

def main():
    config = Config()
    
    # Load prompts
    outline_prompt = load_prompt("outline_prompt")
    writing_prompt = load_prompt("writing_prompt")
    
    # Setup providers (excluding Grok)
    providers = {}
    if config.openai_api_key:
        providers["openai"] = OpenAI(api_key=config.openai_api_key)
    if config.anthropic_api_key:
        providers["anthropic"] = Anthropic(api_key=config.anthropic_api_key)
    if config.gemini_api_key:
        providers["gemini"] = genai.configure(api_key=config.gemini_api_key)
    
    if len(providers) < 3:
        configured_providers = list(providers.keys())
        print(f"Warning: Only {len(providers)} providers configured: {', '.join(configured_providers)}")
    
    # Load data
    ground_truth, redacted_papers = load_data(config)
    
    # Create output directory
    config.output_dir.mkdir(exist_ok=True)
    
    # Run predictions
    print("Starting prediction pipeline...")
    all_predictions = run_predictions(config, providers, redacted_papers, outline_prompt, writing_prompt)
    
    print("Prediction pipeline completed!")
    print(f"Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()








