# Narrative Consistency Testing Framework

This framework tests narrative consistency by comparing character backstories with story content using RAG (Retrieval-Augmented Generation).

## Quick Start

### 1. Pathway RAG Server (Google Drive)

The Pathway server provides a semantic search API that can index documents from Google Drive and local filesystem.

#### Start Google Drive Server

```bash
cd Main_System/Pathway_code
python3 run-server-gdrive.py
```

The server will:
- Index local files from `../data/`
- Index Google Drive folder: `18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0`
- Start HTTP server on `http://127.0.0.1:8745`

#### Kill Server

```bash
# Find and kill process on port 8745
lsof -ti:8745 | xargs kill -9

# Or kill by process name
pkill -f "python3 run-server-gdrive.py"
```

#### Check Server Status

```bash
curl -X POST http://127.0.0.1:8745/v1/statistics \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Query Server

```bash
curl -X POST http://127.0.0.1:8745/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "k": 5}'
```

### 2. Test Framework (`test_framework.py`)

The test framework uses **Pathway RAG server** (Google Drive + Local files) to test narrative consistency.

**Note:** The Pathway server must be running before starting tests!

#### Run Tests

**IMPORTANT:** Make sure the Pathway server is running first!

```bash
# Terminal 1: Start Pathway server
cd Main_System/Pathway_code
python3 run-server-gdrive.py

# Terminal 2: Run tests
cd Main_System

# Run all tests
python3 test_framework.py

# Run specific number of tests
python3 test_framework.py --max-tests 5

# Custom parameters
python3 test_framework.py --k 10 --max-tests 3

# Custom Pathway server location
python3 test_framework.py --pathway-host 127.0.0.1 --pathway-port 8745
```

#### Test Framework Functions

**`NarrativeConsistencyTester`** - Main test class

- `load_test_data()` - Load test cases from CSV file
- `query_pathway_server(question, k=15, character="")` - Query Pathway RAG server
- `check_server_status()` - Verify Pathway server is running
- `process_test_case(test_case, k=15)` - Process single test case:
  - Extract claims from backstory
  - Generate questions from claims
  - Query Pathway server (semantic search)
  - Retrieve relevant chunks
- `run_tests(k=8, max_tests=None)` - Run all tests and evaluate accuracy
- `print_summary(summary)` - Print test results summary

**Workflow:**
1. **Start Pathway server** (must be running first)
2. Load test CSV with character backstories
3. For each test case:
   - Check Pathway server status
   - Extract claims from backstory
   - Generate questions from claims
   - Query Pathway server (semantic search from Google Drive + local files)
   - Retrieve relevant chunks
   - Evaluate consistency
   - Apply decision rule
4. Calculate accuracy

#### Test CSV Format

```csv
id,book_name,char,content,label
1,The Count of Monte Cristo,Edmond Dantes,Edmond was a sailor...,1
2,In Search of Castaways,Captain Grant,Captain Grant was...,0
```

- `id`: Test case ID
- `book_name`: Name of the book (must match files indexed in Pathway server)
- `char`: Character name
- `content`: Backstory content
- `label`: Expected label (`1` = consistent, `0` = inconsistent)

## Components

### RAG Systems

1. **Pathway RAG** (`Pathway_code/`)
   - Semantic-only search (USearch)
   - Google Drive + Local filesystem support
   - HTTP REST API
   - Real-time document updates

2. **Traditional RAG** (`rag.py`)
   - Hybrid search (FAISS + BM25)
   - Character name boosting
   - Local filesystem only
   - **Note:** `test_framework.py` now uses Pathway RAG instead

### Core Modules

- `backstory/claim_extractor.py` - Extracts claims from backstory text
- `questions/question_generator.py` - Generates questions from claims
- `rag.py` - Hybrid RAG system (FAISS + BM25)
- `evaluator.py` - Evaluates consistency between retrieved chunks and backstory
- `test_framework.py` - Main test framework

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

### Google Drive Setup

1. **Service Account**: `service_account.json.json` (in `Main_System/`)
2. **Folder ID**: `18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0`
3. **Share folder** with service account:
   ```
   shubhamkumar28436-gmail-com@speedy-anthem-473507-c0.iam.gserviceaccount.com
   ```

## File Structure

```
Main_System/
├── test_framework.py          # Main test framework
├── rag.py                     # Traditional RAG (hybrid search)
├── evaluator.py               # Consistency evaluator
├── config.py                  # Configuration and API keys
├── backstory/
│   └── claim_extractor.py    # Extract claims from backstory
├── questions/
│   └── question_generator.py # Generate questions from claims
├── Pathway_code/              # Pathway RAG server
│   ├── run-server-gdrive.py  # Start server with Google Drive
│   ├── run-server.py         # Start server (local only)
│   ├── server.py             # Pathway vector store server
│   ├── custom_parser.py      # Document parser
│   └── test_server.py        # Test Pathway server
└── data/                      # Local story files
```

## Commands Reference

### Pathway Server

```bash
# Start with Google Drive
cd Main_System/Pathway_code
python3 run-server-gdrive.py

# Start local only
python3 run-server-gdrive.py  # Edit to comment out Google Drive section

# Kill server
lsof -ti:8745 | xargs kill -9

# Check status
curl -X POST http://127.0.0.1:8745/v1/statistics -H "Content-Type: application/json" -d '{}'

# Query
curl -X POST http://127.0.0.1:8745/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "question", "k": 5}'
```

### Test Framework

```bash
# IMPORTANT: Start Pathway server first!
cd Main_System/Pathway_code
python3 run-server-gdrive.py

# Then run tests (in another terminal)
cd Main_System
python3 test_framework.py

# Run 5 tests
python3 test_framework.py --max-tests 5

# Custom parameters
python3 test_framework.py --k 15 --max-tests 10

# Custom Pathway server
python3 test_framework.py --pathway-host 127.0.0.1 --pathway-port 8745
```

## Output

Test results are saved to `test_results.json` with:
- Test case results
- Evaluations for each question
- Decision rule verdicts
- Accuracy metrics

## Troubleshooting

### Server won't start
- Check if port 8745 is already in use: `lsof -ti:8745`
- Kill existing process: `lsof -ti:8745 | xargs kill -9`

### Google Drive files not appearing
- Verify folder is shared with service account
- Check service account JSON path in `run-server-gdrive.py`
- Wait 1-2 minutes for files to be indexed (streaming mode)

### Test framework errors
- Verify `Books/` directory exists with `.txt` files
- Check `test.csv` format is correct
- Ensure API keys are set in `.env` file

