import os
import requests
import fitz  

# any papers you've manually selected that you want to exclude from the automated selection here
EXCLUDE_PAPERS = {
    
}
SEMANTIC_SCHOLAR_API_TOKEN = ""  # Add your Semantic Scholar API token here
# Dynamically generate set of papers from 'papers' directory
AUTOMATED_PAPERS_FOUND = {os.path.splitext(filename)[0].lower() for filename in os.listdir('papers') if filename.endswith('.pdf')}

# API request details
URL = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk/'
PARAMS = {
    "query": "machine learning | large language models | LLM ",
    "fields": "title,url,venue,publicationVenue,openAccessPdf",
    "publicationTypes": "Conference",
    "year": 2024,
    "openAccessPdf" : "",
    "token": SEMANTIC_SCHOLAR_API_TOKEN
}
HEADERS = {
    "Content-Type": "application/json",
}

DOWNLOAD_HEADERS = { # User-Agent header needed for some sites that block automated GET requests for PDF files-- this makes it think its coming from a browser
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}

PAPER_TARGET = 100
SAVE_DIR = "papers"
os.makedirs(SAVE_DIR, exist_ok=True)

def is_valid_paper(pdf_path) -> bool:
    """check if the PDF has 15 or fewer pages excluding references and appendices."""
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").lower()
            if "references" in text:
                return page_num <= 15  
        return len(doc) <= 15  
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def download_paper(pdf_url, save_path, title, conference) -> bool:
    """download the PDF from the given URL and save it to the specified path."""
    try:
        response = requests.get(pdf_url, headers=DOWNLOAD_HEADERS, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"\nDownloaded: {title}\nConference: {conference}")
        return True
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
        return False

def main():
    response = requests.get(URL, headers=HEADERS, params=PARAMS)
    if response.status_code != 200:
        print(f"Failed to fetch papers: {response.status_code}, {response.text}")
        return

    papers = response.json().get("data", [])
    paper_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.pdf')])

    print(f'Parsing through {len(papers)} papers...')
    
    for paper in papers:
        if paper_count == PAPER_TARGET:
            print("Reached paper target, terminating script...")
            break
        
        pdf_url : str = paper['openAccessPdf']['url']
        title : str = paper['title'].replace('/', '-')
        conference : str = paper['venue']
        
        if not pdf_url:
            print(f"No open access PDF for paper: {title}")
            continue
    
        if title.lower() in MANUALLY_SELECTED_PAPERS:
            print("Already manually selected this paper: ", title)
            continue
    
        if title.lower() in AUTOMATED_PAPERS_FOUND:
            print("Already automated selecting this paper: ", title)
            continue

        save_path = os.path.join(SAVE_DIR, f"{title}.pdf")
        if download_paper(pdf_url, save_path, title, conference):
            if not is_valid_paper(save_path):
                os.remove(save_path)
                print(f"REMOVED: {title} (Does not meet criteria)")
        
        paper_count += 1

if __name__ == "__main__":        
    main()





