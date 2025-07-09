# PaperRAG

A clean, modular RAG (Retrieval-Augmented Generation) system for academic papers built with ChromaDB and OpenAI.

## 📚 Paper Source

This system is designed to work with papers from the **[LLM Agent Papers](https://github.com/zjunlp/LLMAgentPapers)** repository - a comprehensive collection of must-read papers on Large Language Model Agents. The repository contains papers covering:

- 🤖 **Agent Systems**: Personality, Memory, Planning, Tool Use, RL Training
- 🤖💬🤖 **Multi-Agent Systems**: Collaborative, Adversarial, and Casual Conversations  
- 🪐 **Applications**: Real-world agent implementations
- 🖼️ **Frameworks**: Development frameworks and tools
- 🧰 **Resources**: Benchmarks, tool lists, and evaluation metrics

Visit [LLM Agent Papers](https://github.com/zjunlp/LLMAgentPapers) to explore the full collection of papers and resources.

## Features

- **Modular Design**: Clean base classes for easy extension and customization
- **Paper Processing**: Automatic PDF text extraction and chunking with caching
- **Smart Query Enhancement**: Uses OpenAI to improve search queries
- **Academic Focus**: Optimized for research paper Q&A with proper citations
- **Interactive Interface**: User-friendly command-line interface

## Architecture

```
PaperRAG/
├── src/
│   ├── rag/
│   │   ├── __init__.py          # RAG module exports
│   │   ├── base.py              # Base RAG classes
│   │   └── PaperRAG.py          # Paper-specific RAG implementation
│   └── utils/
│       ├── __init__.py          # Utils module exports
│       ├── paper_chunks.py      # PDF processing and chunking
│       └── chroma/              # ChromaDB storage
├── assets/
│   └── papers/                  # Place your PDF papers here
├── main.py                      # Interactive application
└── requirements.txt             # Dependencies
```

## Base Classes

### BaseRAG
Abstract base class for all RAG systems:
- `gen(user_query: str) -> str`: Generate response to user query
- `setup() -> None`: Initialize the RAG system

### ChromaRAG
Base class for ChromaDB-based RAG systems:
- Handles collection management
- Provides query pipeline
- Abstract methods for customization:
  - `_augment_user_query()`: Query enhancement
  - `_generate_answer()`: Answer generation
  - `_load_data()`: Data loading

### PaperRAG
Concrete implementation for academic papers:
- Paper-specific query enhancement
- Academic citation formatting
- Automatic paper chunking and loading

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Add your OpenAI API key to .env
   ```

3. **Add papers**:
   ```bash
   # Place PDF files in assets/papers/
   ```

4. **Run the interactive system**:
   ```bash
   python main.py
   ```

## Usage Examples

### Interactive Mode
```bash
python main.py
```

### Programmatic Usage

```python
from src.rag import PaperRAG
import chromadb

# Initialize
chroma_client = chromadb.PersistentClient(path="chroma")
rag_system = PaperRAG(chroma_client)

# Setup (loads papers and creates collection)
rag_system.setup()

# Ask questions
answer = rag_system.gen("What are the best practices for AI agents?")
print(answer)
```

### Custom RAG System
```python
from src.rag.base import ChromaRAG

class CustomRAG(ChromaRAG):
    def _augment_user_query(self, user_query: str) -> str:
        # Custom query enhancement
        return f"enhanced: {user_query}"
    
    def _generate_answer(self, user_query: str, rag_results) -> str:
        # Custom answer generation
        return "Custom answer based on RAG results"
    
    def _load_data(self) -> None:
        # Custom data loading
        pass
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key

### Paper Processing
- `chunk_size`: Default 800 characters per chunk
- `chunk_overlap`: Default 200 characters overlap
- Cache is automatically managed based on file modifications

### ChromaDB
- Persistent storage in `src/utils/chroma/`
- Collection name: `paper_collection`
- Automatic collection creation and management

## Extending the System

### Adding New RAG Types
1. Extend `ChromaRAG` or `BaseRAG`
2. Implement required abstract methods
3. Add to `src/rag/__init__.py`

### Custom Data Sources
1. Override `_load_data()` method
2. Implement your data loading logic
3. Ensure proper document formatting for ChromaDB

### Custom Query Enhancement
1. Override `_augment_user_query()` method
2. Implement your query improvement logic
3. Return enhanced query string

## File Structure Details

### `src/rag/base.py`
- `BaseRAG`: Abstract base class
- `ChromaRAG`: ChromaDB-specific base class
- Common functionality for all RAG systems

### `src/rag/PaperRAG.py`
- `PaperRAG`: Academic paper RAG implementation
- OpenAI integration for query enhancement and answer generation
- Paper-specific metadata handling

### `src/utils/paper_chunks.py`
- PDF text extraction with PyPDF2
- Intelligent chunking with overlap
- Caching system for performance
- File change detection

### `main.py`
- `PaperRAGApp`: Main application class
- Interactive command-line interface
- Error handling and user experience

## Performance Features

- **Caching**: Paper chunks are cached to avoid reprocessing
- **Incremental Updates**: Only new/modified papers are processed
- **Efficient Storage**: ChromaDB handles vector storage and retrieval
- **Smart Queries**: OpenAI-enhanced search queries for better results

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**:
   - Ensure `.env` file exists with `OPENAI_API_KEY=your_key`

2. **No Papers Found**:
   - Check that PDF files are in `assets/papers/` directory

3. **ChromaDB Errors**:
   - Delete `src/utils/chroma/` directory to reset database
   - Ensure write permissions in the directory

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes `src/` directory

### Debug Mode
Add debug prints to see detailed processing:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the existing class structure
2. Add type hints to all functions
3. Include docstrings for all classes and methods
4. Test with different paper types and queries
5. Update README for new features

## License

This project is open source. Feel free to use and modify for your research needs. 