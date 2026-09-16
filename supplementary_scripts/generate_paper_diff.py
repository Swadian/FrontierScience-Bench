import requests
import sys

EMAIL = "[your email]"

HTML_BOILERPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diff Checker API Testing</title>
</head>
<body>
    
</body>
</html>"""

def get_diff(original: str, redacted: str, paper_title: str) -> str:
    response = requests.post(
        "https://api.diffchecker.com/public/text",
        params={
            "output_type": "html",
            "diff_level": "word",
            "email": EMAIL # 250 requests per month per email
        },
        json={
            "left": original,
            "right": redacted
        }
    )
    code = response.text
    code = f'<h3> {paper_title}</h3>\n' + code
    return code

def create_visualization(diff: str, paper_title: str) -> str:
    # Insert diff before closing body tag
    modified_html = HTML_BOILERPLATE.replace('</body>', f'{diff}\n</body>')
    with open(f'{paper_title[:40]}.html', 'w') as file:
        file.write(modified_html)
        
    return modified_html

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <original_file> <redacted_file>")
        sys.exit(1)
    
    original_file = sys.argv[1]
    redacted_file = sys.argv[2]
    
    original = open(original_file, "r").read()
    redacted = open(redacted_file, "r").read()
    
    diff = get_diff(original, redacted, paper_title="Differentiation of Acute Disseminated Encephalomyelitis from Multiple Sclerosis Using a Novel Brain Lesion Segmentation and Classification Pipeline")
    create_visualization(diff, "Differentiation of Acute Disseminated Encephalomyelitis from Multiple Sclerosis Using a Novel Brain Lesion Segmentation and Classification Pipeline")
    

