import re, urllib.request, ssl, os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create SSL context to handle certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

re_matches = [re.findall(r'https://arxiv\.org/abs/[0-9]{4}\.[0-9]{5}', line) for line in open(os.path.join(script_dir, 'LLMAgentsPapers.md'))]
paper_urls = filter(lambda m: len(m) != 0, re_matches)
paper_pdf_urls = list(map(lambda u: u[0].replace("abs", "pdf"), paper_urls))

# Create papers directory if it doesn't exist
papers_dir = os.path.join(script_dir, 'papers')
os.makedirs(papers_dir, exist_ok=True)

# Download all papers
for i, url in enumerate(paper_pdf_urls):
    try:
        # Extract paper ID from URL for filename
        paper_id = url.split('/')[-1]
        filename = os.path.join(papers_dir, f"{paper_id}.pdf")
        
        print(f"Downloading {i+1}/{len(paper_pdf_urls)}: {paper_id}")
        
        # Use SSL context for the download
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
        print(f"✓ Downloaded: {filename}")
        
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")

print(f"\nDownload complete! Downloaded {len(paper_pdf_urls)} papers to the papers/ folder.")
