import glob
import pprint

import PyPDF2
import os
import json
import hashlib
from typing import Dict, List
from collections import defaultdict

import chromadb
import openai
from chromadb import QueryResult

from dotenv import load_dotenv
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from .logger import get_logger

PATH_TO_PAPERS = "assets/papers"
CACHE_FILE = "assets/paper_chunks_cache.json"

def papers_to_chunks(chunk_size=800, chunk_overlap=200) -> Dict[str, List[str]]:
    logger = get_logger("PaperRAG.paper_chunks")
    
    # Check if cache exists and is valid
    cache_data = _load_cache(chunk_size, chunk_overlap)
    if cache_data:
        logger.info("Using cached chunks - no recomputation needed!")
        return cache_data
    
    logger.info("Cache not found or invalid - processing PDFs...")
    out = defaultdict(list)
    for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
        try:
            logger.info(f"Reading: {os.path.basename(file)}")
            
            # Open PDF file in binary mode
            with open(file, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                logger.info(f"Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
                
                # Create chunks with overlap
                chunks = []
                i = 0
                while i < len(text):
                    chunk_start = i
                    chunk_end = min(i + chunk_size, len(text))
                    
                    # Extract the chunk
                    chunk = text[chunk_start:chunk_end]
                    chunks.append(chunk)
                    
                    # Move to next chunk with overlap
                    i += (chunk_size - chunk_overlap)
                
                # Store chunks in output dictionary
                out[os.path.basename(file)] = chunks

        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    result = dict(out)
    
    # Save to cache
    _save_cache(result, chunk_size, chunk_overlap)
    
    return result


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of file modification time and size"""
    stat = os.stat(filepath)
    # Combine modification time and file size for hash
    content = f"{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_cache(chunk_size: int, chunk_overlap: int) -> Dict[str, List[str]] | None:
    """Load cached chunks if valid"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        # Check if cache parameters match
        if cache.get('chunk_size') != chunk_size or cache.get('chunk_overlap') != chunk_overlap:
            return None
        
        # Check if all files in cache still exist and haven't changed
        file_hashes = cache.get('file_hashes', {})
        for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
            filename = os.path.basename(file)
            current_hash = get_file_hash(file)
            
            if filename not in file_hashes or file_hashes[filename] != current_hash:
                return None
        
        return cache.get('chunks', {})
        
    except Exception as e:
        logger = get_logger("PaperRAG.paper_chunks")
        logger.error(f"Error loading cache: {e}")
        return None


def _save_cache(chunks: Dict[str, List[str]], chunk_size: int, chunk_overlap: int):
    """Save chunks to cache with file hashes"""
    # Calculate file hashes
    file_hashes = {}
    for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
        filename = os.path.basename(file)
        file_hashes[filename] = get_file_hash(file)
    
    cache_data = {
        'chunks': chunks,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'file_hashes': file_hashes
    }
    
    logger = get_logger("PaperRAG.paper_chunks")
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Cache saved to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


if __name__ == "__main__":
    # Test the paper chunking functionality
    load_dotenv()
    paper_chunks = papers_to_chunks()
    logger = get_logger("PaperRAG.paper_chunks")
    logger.info(f"Processed {len(paper_chunks)} papers")
    for paper, chunks in paper_chunks.items():
        logger.info(f"  {paper}: {len(chunks)} chunks")