# Narrative Consistency Testing Framework

A framework for testing narrative consistency by comparing character backstories with story content using Pathway RAG (Retrieval-Augmented Generation) with Google Drive integration.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Server Management](#server-management)
- [Running Tests](#running-tests)
- [Output Files](#output-files)
- [Configuration](#configuration)

## Overview

This framework:
1. Uses **Pathway RAG Server** to index and search through novels from Google Drive
2. Generates RAG-friendly questions from character backstories using Gemini
3. Retrieves relevant story chunks using semantic search
4. Evaluates consistency using Gemini to compare backstory claims with story content
5. Produces a `results.csv` file with predictions and AI-generated rationales

## Quick Start

### 1. Install Dependencies

```bash
cd Main_System
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the `Main_System` directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json.json
```

**Note:** The Google Drive service account credentials file (`service_account.json.json`) should be placed in the `Main_System` directory.

### 3. Start the Pathway Server

**Option A: Using the helper script (Recommended)**

```bash
./start_server.sh
```

**Option B: Manual start**

```bash
cd Pathway_code
python3 run-server-gdrive.py
```

The server will:
- Index documents from Google Drive folder: `18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0`
- Start HTTP server on `http://127.0.0.1:8745`
- Keep running until you stop it (Ctrl+C)

### 4. Run Tests

**In a new terminal:**

```bash
cd Main_System
python3 test_framework.py
```

**Run specific number of tests:**
```bash
python3 test_framework.py --max-tests 5
```

**Custom parameters:**
```bash
python3 test_framework.py --k 10 --max-tests 3 --hybrid-search
```

### 5. Stop the Server (when done)

**Option A: Using the helper script**

```bash
./kill_server.sh
```

**Option B: Manual stop**

```bash
# Kill by port
lsof -ti:8745 | xargs kill -9

# Or kill by process name
pkill -f "python3.*run-server-gdrive.py"
```

## Installation

### Requirements

- Python 3.8 or higher
- Google Gemini API key
- Google Cloud service account credentials (for Google Drive access)

### Step-by-Step Setup

1. **Clone the repository** (if applicable)

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Gemini API:**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add to `.env` file: `GEMINI_API_KEY=your_key_here`

4. **Set up Google Drive access:**
   - Create a Google Cloud service account
   - Download the JSON credentials file
   - Save as `service_account.json.json` in `Main_System` directory
   - Share your Google Drive folder with the service account email
   - Add to `.env`: `GOOGLE_APPLICATION_CREDENTIALS=service_account.json.json`

5. **Verify test data:**
   - Ensure `test.csv` exists in the parent directory (`../test.csv`)
   - Ensure the Google Drive folder contains the novel text files

## Server Management

### Starting the Server

The Pathway server must be running before running tests.

**Automatically (Recommended):**
```bash
./start_server.sh
```

**Manually:**
```bash
cd Pathway_code
python3 run-server-gdrive.py
```

### Checking Server Status

```bash
curl -X POST http://127.0.0.1:8745/v1/statistics \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Stopping the Server

**Automatically (Recommended):**
```bash
./kill_server.sh
```

**Manually:**
```bash
lsof -ti:8745 | xargs kill -9
```

## Running Tests

### Basic Usage

```bash
python3 test_framework.py
```

### Command-Line Options

- `--test-csv PATH`: Path to test CSV file (default: `../test.csv`)
- `--k N`: Number of chunks to retrieve per question (default: 10)
- `--max-tests N`: Maximum number of tests to run (default: all)
- `--pathway-host HOST`: Pathway server host (default: 127.0.0.1)
- `--pathway-port PORT`: Pathway server port (default: 8745)
- `--hybrid-search`: Enable hybrid search (BM25 + semantic) for better accuracy

### Examples

```bash
# Run all tests with default settings
python3 test_framework.py

# Run first 5 tests
python3 test_framework.py --max-tests 5

# Run with hybrid search and more chunks
python3 test_framework.py --k 15 --hybrid-search --max-tests 10

# Use custom Pathway server location
python3 test_framework.py --pathway-host 127.0.0.1 --pathway-port 8745
```

## Output Files

After running tests, the framework generates:

1. **`results.csv`**: Submission file with format:
   - `Story ID`: Test case ID
   - `Prediction`: 1 (consistent) or 0 (inconsistent)
   - `Rationale`: AI-generated explanation for the verdict

2. **`test_results.json`**: Detailed results with:
   - All generated questions
   - Retrieved chunks for each question
   - Individual evaluation results
   - Final verdict and decision rule

## Configuration

### Environment Variables

Create a `.env` file in `Main_System` directory:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional (if using absolute path for service account)
GOOGLE_APPLICATION_CREDENTIALS=service_account.json.json
```

### Pathway Server Configuration

Edit `Pathway_code/run-server-gdrive.py` to change:
- Google Drive folder ID
- Server host/port
- Embedding model
- Chunk size/overlap

### Test Framework Configuration

Edit `test_framework.py` to change:
- Default number of chunks (`k`)
- Evaluation parameters
- Decision rule logic

## Architecture

1. **Pathway RAG Server**: Indexes documents from Google Drive, provides semantic search API
2. **Question Generator**: Uses Gemini to generate RAG-friendly questions from backstories
3. **Retrieval**: Queries Pathway server for relevant story chunks
4. **Evaluation**: Uses Gemini to evaluate consistency between backstory claims and retrieved chunks
5. **Decision Rule**: Aggregates evaluations into final verdict (CONSISTENT/INCONSISTENT)

## Troubleshooting

### Server won't start
- Check if port 8745 is already in use: `lsof -ti:8745`
- Kill existing process: `./kill_server.sh`
- Check Google Drive credentials and folder sharing

### Server connection timeout
- Ensure server is running: Check with `curl` command above
- Verify host/port settings match server configuration
- Check firewall settings

### No results retrieved
- Verify Google Drive folder is shared with service account
- Check that documents are indexed (use `/v1/statistics` endpoint)
- Review server logs for errors

### API errors
- Verify Gemini API key is set correctly in `.env`
- Check API quota/limits
- Ensure service account credentials are valid

## File Structure

```
Main_System/
├── test_framework.py          # Main test framework
├── evaluator.py               # Consistency evaluation logic
├── questions/
│   └── question_generator.py  # Question generation using Gemini
├── Pathway_code/
│   ├── run-server-gdrive.py   # Server startup script
│   ├── server.py              # Pathway VectorStoreServer
│   ├── custom_parser.py       # Document parser
│   └── hybrid_retriever.py    # Hybrid search (optional)
├── config.py                  # Configuration and API key loading
├── start_server.sh            # Helper script to start server
├── kill_server.sh             # Helper script to kill server
├── requirements.txt           # Python dependencies
├── results.csv                # Output file (generated)
└── README.md                  # This file
```

## License

[Add your license here]

## Contact

[Add contact information here]
