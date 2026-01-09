# Long-Context Consistency of LLM - KDSH 26

Binary classification system for verifying the global consistency of hypothetical character backstories against long-form narrative texts.

## Overview

This system checks if character backstories are consistent with the actual story content in novels using a RAG (Retrieval-Augmented Generation) approach. It:

1. **Extracts claims** from character backstories
2. **Generates questions** from those claims
3. **Retrieves relevant chunks** from the story using hybrid search (FAISS + BM25)
4. **Evaluates consistency** by comparing retrieved story content with backstory facts

## Project Structure

```
.
├── Main System/              # Main source code
│   ├── backstory/           # Claim extraction from backstories
│   │   └── claim_extractor.py
│   ├── questions/           # Question generation
│   │   └── question_generator.py
│   ├── models/              # Data models and schemas
│   │   └── schemas.py
│   ├── data/                # Sample data
│   │   ├── sample_row.json
│   │   └── sample_story.txt
│   ├── config.py            # Configuration and API keys
│   ├── evaluator.py         # Consistency evaluation
│   ├── rag.py               # RAG class implementation
│   ├── main.py              # Main pipeline
│   ├── test_framework.py    # Test framework
│   └── requirements.txt     # Python dependencies
│   ├── test_llm.py          # LLM testing utilities
│   └── requirements.txt     # Python dependencies
├── Books/                   # Novel text files
│   ├── In search of the castaways.txt
│   └── The Count of Monte Cristo.txt
├── test.csv                 # Test cases with backstories
├── train.csv                # Training data (optional)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shubham21-ai/Long-Context-Consistency-of-LLM-KDSH-26.git
cd Long-Context-Consistency-of-LLM-KDSH-26
```

2. Install dependencies:
```bash
cd "Main System"
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the `Main System/` directory with:
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Main Pipeline
Run the main pipeline on sample data:
```bash
cd "Main System"
python main.py
```

### Test Framework
Run the test framework on test.csv:
```bash
cd "Main System"
python test_framework.py
```

## How It Works

### 1. Claim Extraction
The system extracts explicit events/claims from character backstories using structured extraction (subject-relation-object format).

### 2. Question Generation
Each claim is converted into 2-3 specific, verifiable questions that can be answered by searching the story text.

### 3. RAG Retrieval
- **Hybrid Search**: Combines semantic search (FAISS) and keyword search (BM25)
- **Embeddings**: Uses sentence-transformers for semantic embeddings
- **Retrieval**: Returns top-k relevant chunks per question

### 4. Evaluation
- Compares retrieved story chunks with backstory claims
- Classifies as: CONSISTENT, INCONSISTENT, or UNCERTAIN
- Applies decision rules to aggregate results

## Components

### Claim Extractor (`backstory/claim_extractor.py`)
Extracts structured events from backstory text using LLM.

### Question Generator (`questions/question_generator.py`)
Generates specific questions from extracted claims.

### RAG System (`rag.py`)
Hybrid retrieval system combining:
- FAISS for semantic similarity
- BM25 for keyword matching
- Sentence transformers for embeddings

### Evaluator (`evaluator.py`)
Evaluates consistency between backstory claims and story content.

### Test Framework (`test_framework.py`)
End-to-end testing framework that:
- Processes novels from Books folder
- Loads test cases from CSV
- Runs consistency checks
- Generates accuracy metrics

## Requirements

- Python 3.8+
- Groq API key

## Models Used

- **Evaluation / LLM**: Qwen 3-32B (via Groq)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)

## License

This project is for research purposes.

