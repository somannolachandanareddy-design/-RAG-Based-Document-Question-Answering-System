
# RAG-Based Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers user queries by retrieving relevant information from documents using vector similarity search and generating context-aware responses using large language models.

## Features

- Document ingestion and preprocessing pipeline
- Semantic search using vector embeddings
- Context retrieval with FAISS
- LLM-powered answer generation
- REST API for question-answering workflows
- Automated evaluation using RAGAS metrics
- Support for multiple LLM providers

## Tech Stack

- Python
- FastAPI
- LangChain
- Sentence-BERT
- FAISS
- OpenAI GPT / Gemini
- RAGAS

## Workflow

Documents → Embedding Generation → Vector Store → Retrieval → LLM Generation → Response

##Future Improvements:
Hybrid search (keyword + vector retrieval)
Citation-aware responses
Multi-document conversation support
Role-based document access
