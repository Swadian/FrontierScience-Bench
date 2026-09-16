import fitz
import requests
import pathlib

PAPER_LINKS = './paper_links.txt'

def download_papers():
    """ Downloads papers in PDF format from paper_links.txt file and stores
        them in ./downloads.""" 
        
    with open(PAPER_LINKS, 'r') as file:
        paper_links = file.readlines()

    for index, link in enumerate(paper_links):
        if not link.strip():
            continue
        try:
            response = requests.get(link.strip())
            response.raise_for_status()
            with open(f'./downloads/paper{index}.pdf', "wb") as file:
                file.write(response.content)
            print(f'Downloaded paper {index}')
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e} while downloading paper {index}")

def process_papers():
    """ Uses pymupdf/fitz to parse each PDF file in downloads into a text file,
        stored in ./downloads_txt. """
        
    downloads = pathlib.Path('./downloads')
    files = [f.name for f in downloads.rglob("*") if f.is_file()]
    
    for index, file in enumerate(files):
        doc = fitz.open(f'downloads/{file}')
        with open(f'./downloads_txt/{file}.txt', 'w') as file:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text : str = page.get_text()
                reference_index = text.find("References")
                if reference_index != -1: # remove text from references
                    text = text[:reference_index]
                    file.write(text)
                    break # all following pages will also be references so stop iteration
                
                file.write(text)
                
        print(f'Parsed paper{index}.pdf into text file.')
    

if __name__ == "__main__":
    download_papers()
    process_papers()


