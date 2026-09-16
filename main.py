#!/usr/bin/env python3
import os
import sys
import json
import time
import pickle
import difflib
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated

from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from pydantic import BaseModel, Field

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

from verdict import Pipeline, Layer
from verdict.common.judge import CategoricalJudgeUnit
from verdict.scale import DiscreteScale
from verdict.schema import Schema
from verdict.transform import MaxPoolUnit
from verdict.util import ratelimit

load_dotenv()


class HumanRedactedPaperUpdated:
    def __setstate__(self, state):
        payload = state.get("__dict__", state) if isinstance(state, dict) else state
        self.__dict__.update(payload)


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
    grok_model: str = "xai/grok-2"

    ground_truth_path: Path = Path("data/ground_truth.pkl")
    redacted_papers_path: Path = Path("data/redacted_papers.pkl")
    alignment_cache_path: Path = Path("data/alignment.json")
    prompt_dir: Path = Path("prompts")
    predictions_dir: Path = Path("results/predictions")
    judging_dir: Path = Path("results/judging")

    num_tries: int = 1          # outline/write refinement iterations per paper
    judge_repeats: int = 5      # repeated judge calls per paper, reduced by majority vote
    intro_match_threshold: float = 0.7
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0

    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")

PREDICTOR_NAMES = ["gpt_4o", "o3_mini", "claude", "gemini"]
JUDGE_NAMES = ["gpt_4o", "o3_mini", "claude", "grok"]

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

def load_prompt(config: Config, filename: str) -> str:
    with open(config.prompt_dir / f"{filename}.txt", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------------------------------------------------------
# Structured-output schemas
# -----------------------------------------------------------------------------

class Section(BaseModel):
    name: str
    content: str


class Outline(BaseModel):
    proposed_method: str = Field(..., description=(
        "Using the given information, first provide inspiration behind a new proposed method to "
        "address the main research problem. You should also motivate why the proposed method would "
        "work better than existing works. Then, explain how the proposed approach works, and describe "
        "all the essential steps. Do NOT repeat proposed methods that are already in the "
        "'attempted_methods.'"
    ))
    experimental_plan: str = Field(..., description=(
        "Break down EVERY single step in 'proposed_method'. Every step MUST be executable. Cover ALL "
        "essential details such as the datasets, models, metrics to be used, etc."
    ))


class Contributions(BaseModel):
    contributions: List[Section] = Field(
        ..., description="The contributions section will include ALL of the following sections: Methods, Experiments."
    )


class GeminiSection(TypedDict):
    name: str
    content: str


class GeminiOutline(TypedDict):
    proposed_method: Annotated[str, (
        "Using the given information, first provide inspiration behind a new proposed method to "
        "address the main research problem. You should also motivate why the proposed method would "
        "work better than existing works. Then, explain how the proposed approach works, and describe "
        "all the essential steps. Do NOT repeat proposed methods that are already in the "
        "'attempted_methods.'"
    )]
    experimental_plan: Annotated[str, (
        "Break down EVERY single step in 'proposed_method'. Every step MUST be executable. Cover ALL "
        "essential details such as the datasets, models, metrics to be used, etc."
    )]


class GeminiContributions(TypedDict):
    contributions: Annotated[
        List[GeminiSection],
        "The contributions section will include ALL of the following sections: Methods, Experiments.",
    ]


# -----------------------------------------------------------------------------
# Data loading + redacted/ground-truth alignment
#
# data/ground_truth.pkl holds the final 88 curated papers. data/redacted_papers.pkl
# is a superset of 101 papers predating a later copyright-driven trim, in the
# same relative order but missing 13 entries. There's no shared id between the
# two files, so papers are realigned by fuzzy-matching the (unredacted) start
# of each introduction. See conversation history for how this was verified.
# -----------------------------------------------------------------------------

def _intro_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a[:400], b[:400]).ratio()


def build_alignment(ground_truths: List[dict], redacted: List[Any], threshold: float) -> List[int]:
    """Return, for each ground-truth index i, the matching index into `redacted`."""
    mapping: List[int] = []
    seen = set()
    for i, gt in enumerate(ground_truths):
        scores = sorted(
            ((_intro_similarity(gt["introduction"], r.introduction), j) for j, r in enumerate(redacted)),
            reverse=True,
        )
        best_score, best_j = scores[0]
        if best_score < threshold:
            raise ValueError(f"No confident redacted-paper match for ground_truth[{i}] (best score {best_score:.2f})")
        if best_j in seen:
            raise ValueError(f"redacted_papers[{best_j}] matched more than one ground_truth entry")
        seen.add(best_j)
        mapping.append(best_j)
    if mapping != sorted(mapping):
        raise ValueError("Alignment is not order-preserving; refusing to use it")
    return mapping


def load_dataset(config: Config) -> List[Dict[str, Any]]:
    """Returns a list of {paper_id, ground_truth, redacted_input} dicts, aligned and cached."""
    with open(config.ground_truth_path, "rb") as f:
        ground_truths = [json.loads(s) for s in pickle.load(f)]

    with open(config.redacted_papers_path, "rb") as f:
        redacted = pickle.load(f)

    if config.alignment_cache_path.exists():
        alignment = json.loads(config.alignment_cache_path.read_text())
    else:
        alignment = build_alignment(ground_truths, redacted, config.intro_match_threshold)
        config.alignment_cache_path.parent.mkdir(parents=True, exist_ok=True)
        config.alignment_cache_path.write_text(json.dumps(alignment))

    dataset = []
    for i, gt in enumerate(ground_truths):
        r = redacted[alignment[i]]
        redacted_input = json.dumps({
            "introduction": r.introduction,
            "related_works": r.related_works,
        })
        dataset.append({
            "paper_id": f"paper_{i:03d}",
            "ground_truth": gt,
            "redacted_input": redacted_input,
        })
    return dataset


# -----------------------------------------------------------------------------
# Per-provider API calls
# -----------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    return model.startswith("o1") or model.startswith("o3")


def api_call_openai(client, model, system_prompt, inputs, response_format, temperature=0.0):
    messages = [{"role": "system", "content": system_prompt}]
    for user_input in inputs:
        messages.append({"role": "user", "content": user_input})

    kwargs = dict(model=model, messages=messages, response_format=response_format)
    if not _is_reasoning_model(model):
        kwargs["temperature"] = temperature

    response = client.beta.chat.completions.parse(**kwargs)
    return response.choices[0].message.content


def api_call_anthropic(client, model, system_prompt, inputs, temperature=0.0, max_tokens=8192):
    messages = [{"role": "user", "content": text} for text in inputs]
    message = client.messages.create(
        model=model,
        system=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return message.content[0].text


def api_call_gemini(model_client, messages, temperature=0.8, max_tokens=4096):
    combined_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    response = model_client.generate_content(
        combined_text,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=model_client._response_schema,
        ),
    )
    return response.text


# -----------------------------------------------------------------------------
# Predictor factories: each returns (outline_fn, write_fn), both
# `str -> str` over JSON text, so the calling loop can treat every provider
# identically.
# -----------------------------------------------------------------------------

def make_openai_predictor(client, model, outline_prompt, writing_prompt):
    def do_outline(redacted_input_json: str, attempted_methods_json: str) -> str:
        return api_call_openai(client, model, outline_prompt, [redacted_input_json, attempted_methods_json], Outline, temperature=0.8)

    def do_write(outline_json: str) -> str:
        return api_call_openai(client, model, writing_prompt, [outline_json], Contributions, temperature=0.8)

    return do_outline, do_write


def make_claude_predictor(client, model, outline_prompt, formatting_outline_prompt, writing_prompt):
    combined_outline_prompt = outline_prompt + "\n\n" + formatting_outline_prompt

    def do_outline(redacted_input_json: str, attempted_methods_json: str) -> str:
        return api_call_anthropic(client, model, combined_outline_prompt, [redacted_input_json, attempted_methods_json], temperature=0.8)

    def do_write(outline_json: str) -> str:
        return api_call_anthropic(client, model, writing_prompt, [outline_json], temperature=0.8)

    return do_outline, do_write


def make_gemini_predictor(model_name, outline_prompt, writing_prompt):
    outline_model = genai.GenerativeModel(model_name=model_name, system_instruction=outline_prompt)
    outline_model._response_schema = GeminiOutline
    write_model = genai.GenerativeModel(model_name=model_name, system_instruction=writing_prompt)
    write_model._response_schema = GeminiContributions

    def do_outline(redacted_input_json: str, attempted_methods_json: str) -> str:
        messages = [
            {"role": "user", "content": redacted_input_json},
            {"role": "assistant", "content": attempted_methods_json},
        ]
        return api_call_gemini(outline_model, messages, temperature=0.8)

    def do_write(outline_json: str) -> str:
        return api_call_gemini(write_model, [{"role": "user", "content": outline_json}], temperature=0.8)

    return do_outline, do_write


def setup_predictors(config: Config, prompts: Dict[str, str]) -> Dict[str, Tuple[Any, Any]]:
    missing = [k for k in ("openai_api_key", "anthropic_api_key", "gemini_api_key")
               if not getattr(config, k)]
    if missing:
        raise ValueError(f"Missing required API keys for prediction: {missing}")

    openai_client = OpenAI(api_key=config.openai_api_key)
    anthropic_client = Anthropic(api_key=config.anthropic_api_key)
    genai.configure(api_key=config.gemini_api_key)

    outline_prompt = prompts["outline"]
    formatting_outline_prompt = prompts["formatting_outline"]
    writing_prompt = prompts["writing"]

    return {
        "gpt_4o": make_openai_predictor(openai_client, config.gpt_model_4o, outline_prompt, writing_prompt),
        "o3_mini": make_openai_predictor(openai_client, config.gpt_model_o3_mini, outline_prompt, writing_prompt),
        "claude": make_claude_predictor(anthropic_client, config.claude_model, outline_prompt, formatting_outline_prompt, writing_prompt),
        "gemini": make_gemini_predictor(config.gemini_model, outline_prompt, writing_prompt),
    }


# -----------------------------------------------------------------------------
# Retry helper
# -----------------------------------------------------------------------------

def with_retries(fn, *args, max_retries=3, backoff_seconds=5.0, **kwargs):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad; caller logs and skips
            last_error = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    raise last_error


# -----------------------------------------------------------------------------
# Prediction pipeline
# -----------------------------------------------------------------------------

def run_predictions(config: Config, predictors: Dict[str, Tuple[Any, Any]], dataset: List[Dict[str, Any]],
                     provider_names: List[str], force: bool = False) -> None:
    for provider in provider_names:
        do_outline, do_write = predictors[provider]
        provider_dir = config.predictions_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)

        for paper in tqdm(dataset, desc=f"Predicting [{provider}]"):
            paper_dir = provider_dir / paper["paper_id"]
            paper_dir.mkdir(parents=True, exist_ok=True)
            contributions_path = paper_dir / f"contributions_{config.num_tries}.json"

            if contributions_path.exists() and not force:
                continue

            try:
                attempted_methods: List[str] = []
                outline_text = None
                for i in range(config.num_tries):
                    attempted_json = json.dumps({"attempted_methods": attempted_methods})
                    outline_text = with_retries(
                        do_outline, paper["redacted_input"], attempted_json,
                        max_retries=config.max_retries, backoff_seconds=config.retry_backoff_seconds,
                    )
                    (paper_dir / f"outline_{i + 1}.json").write_text(outline_text)
                    attempted_methods.append(json.loads(outline_text)["proposed_method"])

                    contributions_text = with_retries(
                        do_write, outline_text,
                        max_retries=config.max_retries, backoff_seconds=config.retry_backoff_seconds,
                    )
                    (paper_dir / f"contributions_{i + 1}.json").write_text(contributions_text)
            except Exception as e:
                print(f"[{provider}] {paper['paper_id']} failed after retries: {e}", file=sys.stderr)
                continue


def load_predictions(config: Config, provider: str, dataset: List[Dict[str, Any]]) -> Dict[str, str]:
    """paper_id -> final contributions JSON text (final try only), skipping missing ones."""
    provider_dir = config.predictions_dir / provider
    predictions = {}
    for paper in dataset:
        path = provider_dir / paper["paper_id"] / f"contributions_{config.num_tries}.json"
        if path.exists():
            predictions[paper["paper_id"]] = path.read_text()
    return predictions


# -----------------------------------------------------------------------------
# Judging pipeline (Verdict), adapted from notebooks/Verdict_Framework_Testing.ipynb
# -----------------------------------------------------------------------------

def configure_judge_env(config: Config) -> None:
    # litellm (used internally by verdict) reads provider credentials from env vars.
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    if config.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = config.gemini_api_key
    if config.grok_api_key:
        os.environ["XAI_API_KEY"] = config.grok_api_key
        os.environ["XAI_BASE_URL"] = "https://api.x.ai/v1"


JUDGE_DISPLAY_NAMES = {
    "gpt_4o": "GPT-4o",
    "o3_mini": "o3-mini",
    "claude": "Claude 3.5",
    "grok": "Grok 2",
}


def judge_model_ids(config: Config) -> Dict[str, str]:
    return {
        "gpt_4o": config.gpt_model_4o,
        "o3_mini": config.gpt_model_o3_mini,
        "claude": config.claude_model,
        "grok": config.grok_model,
    }


def build_judge_pipeline(config: Config, judge_prompt: str, judges: List[str]) -> Tuple[Pipeline, List[str]]:
    score_scale = DiscreteScale(list(range(1, 11)))
    model_ids = judge_model_ids(config)

    def make_layer(model_name: str):
        judge = (
            CategoricalJudgeUnit(categories=score_scale, explanation=True)
            .prompt(judge_prompt + """
            Ground truth methodology:
            {source.ground_truth}

            Predicted methodology:
            {source.predicted}
            """)
            .via(model_name, retries=3, temperature=0.7)
        )
        return Layer(judge, repeat=config.judge_repeats) >> MaxPoolUnit()

    included = [j for j in judges]
    layers = [make_layer(model_ids[j]) for j in included]
    pipeline = Pipeline("LLM Majority Judge") >> Layer(layers)
    return pipeline, included


def run_judging(config: Config, judge_prompt: str, dataset: List[Dict[str, Any]], force: bool = False) -> None:
    configure_judge_env(config)
    ratelimit.disable()

    for provider in PREDICTOR_NAMES:
        predictions = load_predictions(config, provider, dataset)
        if not predictions:
            print(f"[{provider}] no predictions found, skipping judging", file=sys.stderr)
            continue

        judges = [j for j in JUDGE_NAMES if j != provider]  # self-omission
        pipeline, included = build_judge_pipeline(config, judge_prompt, judges)
        included_display = [JUDGE_DISPLAY_NAMES[j] for j in included]

        out_dir = config.judging_dir / provider
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "aggregated_results.csv"
        already_done = set()
        if csv_path.exists():
            already_done = set(pd.read_csv(csv_path)["paper"])

        for paper in tqdm(dataset, desc=f"Judging [{provider}]"):
            paper_id = paper["paper_id"]
            if paper_id not in predictions or (paper_id in already_done and not force):
                continue

            try:
                ground_truth = str(paper["ground_truth"]["methodology"])
                prediction = str(json.loads(predictions[paper_id])["contributions"])
            except Exception as e:
                print(f"[{provider}] {paper_id}: could not extract contributions: {e}", file=sys.stderr)
                continue

            content = Schema.of(ground_truth=ground_truth, predicted=prediction)

            try:
                judge_dict, key_list = with_retries(
                    pipeline.run, content,
                    max_retries=config.max_retries, backoff_seconds=config.retry_backoff_seconds,
                )
            except Exception as e:
                print(f"[{provider}] {paper_id}: judging failed after retries: {e}", file=sys.stderr)
                continue

            mode_scores = {}
            mode_explanations = {}
            key_idx = 0
            for name in included_display:
                mode_scores[name] = judge_dict[key_list[key_idx]]
                mode_explanations[name] = judge_dict[key_list[key_idx + 1]]
                key_idx += 2

            aggregated_score = sum(mode_scores.values()) / len(included_display)

            paper_out_dir = out_dir / paper_id
            paper_out_dir.mkdir(parents=True, exist_ok=True)
            for name in included_display:
                (paper_out_dir / f"{name}.txt").write_text(str(mode_explanations[name]))

            row = {"paper": paper_id, **mode_scores, "aggregated": aggregated_score}
            df = pd.DataFrame([row])[["paper"] + included_display + ["aggregated"]]
            df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FrontierScience-Bench prediction + judging pipeline")
    parser.add_argument("--stage", choices=["predict", "judge", "all"], default="all")
    parser.add_argument("--predictors", default=",".join(PREDICTOR_NAMES),
                         help="Comma-separated subset of predictors to run")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N papers")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs already exist")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    prompts = {
        "outline": load_prompt(config, "outline_prompt"),
        "formatting_outline": load_prompt(config, "formatting_outline_prompt"),
        "writing": load_prompt(config, "writing_prompt"),
        "judge": load_prompt(config, "judge_prompt"),
    }

    print("Loading and aligning dataset...")
    dataset = load_dataset(config)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} papers.")

    if args.stage in ("predict", "all"):
        provider_names = [p.strip() for p in args.predictors.split(",") if p.strip()]
        print(f"Starting prediction pipeline for: {provider_names}")
        predictors = setup_predictors(config, prompts)
        run_predictions(config, predictors, dataset, provider_names, force=args.force)
        print("Prediction pipeline completed.")

    if args.stage in ("judge", "all"):
        print("Starting judging pipeline...")
        run_judging(config, prompts["judge"], dataset, force=args.force)
        print("Judging pipeline completed.")

    print(f"Results saved to: {config.predictions_dir} and {config.judging_dir}")


if __name__ == "__main__":
    main()
