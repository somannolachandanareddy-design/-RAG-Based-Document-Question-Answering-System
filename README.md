
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

Step1 : Upload Documents then ask what you want relevant from documents
<img width="1361" height="605" alt="image" src="https://github.com/user-attachments/assets/6c4e2702-ab86-44c5-818b-9616bcc22557" />
<img width="954" height="458" alt="image" src="https://github.com/user-attachments/assets/6e7ed342-d741-4140-b31b-b88c9e648e73" />
It Explains what we want simply Instead of Reading Entire Document
Examples To this RAG:
Example 1: Resume Assistant

Upload:

Your resume

Ask:

"Can I apply for a Google SDE role?"
"What skills are missing for a backend developer role?"
"Summarize my projects."

This is exactly the test you performed.

Example 2: Student Notes Assistant

Upload:

DBMS notes
OS notes
CN notes

Ask:

"Explain deadlock."
"Give important exam questions from Unit 3."
"Summarize chapter 5."

Useful for students.

Example 3: Research Paper Assistant

Upload:

AI/ML research papers

Ask:

"What is the main contribution of this paper?"
"What are the limitations?"
"Compare this paper with another."

Useful for research and higher studies.

Example 4: Company Policy Chatbot

Upload:

HR policy
Leave policy
Employee handbook

Ask:

"How many sick leaves are allowed?"
"What is the notice period?"
"What are the work-from-home rules?"

Very common enterprise use case.

Example 5: Legal Document Assistant

Upload:

Contracts
Agreements

Ask:

"What is the termination clause?"
"What penalties are mentioned?"
"When does the contract expire?"
Example 6: Medical Document Assistant

Upload:

Medical guidelines
Research reports

Ask:

"What treatments are recommended?"
"What are the symptoms listed?"
Example 7: Interview Preparation Assistant

Upload:

DSA notes
Java interview PDFs

Ask:

"Generate interview questions from these notes."
"Explain multithreading."
Example 8: Product Manual Assistant

Upload:

User manuals
Technical documentation

Ask:

"How do I reset the device?"
"What does error code E45 mean?"

Companies use this a lot.

Example 9: College Circular Assistant

Upload:

University notices
Circulars

Ask:

"What is the registration deadline?"
"Which documents are required?"
