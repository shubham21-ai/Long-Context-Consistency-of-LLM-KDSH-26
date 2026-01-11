# Narrative Consistency Testing Framework

A framework for testing narrative consistency by comparing character backstories with story content using Pathway RAG (Retrieval-Augmented Generation) with Google Drive integration.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Google Cloud service account credentials (for Google Drive access)

### Installation

1. **Install dependencies:**
   ```bash
   cd Main_System
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   
   Create a `.env` file in the `Main_System` directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=service_account.json.json
   ```
   
   **Get your Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

3. **Set up Google Drive:**
   - Create a Google Cloud service account
   - Download the JSON credentials file
   - Save as `service_account.json.json` in the `Main_System` directory
   - Share your Google Drive folder (with novels) with the service account email
   - Share your CSV file (`test.csv` or `train.csv`) with the service account email

## Running the Framework

### Step 1: Start the Pathway Server

**Option A: Using the helper script (Recommended)**
```bash
cd Main_System
./start_server.sh
```

**Option B: Manual start**
```bash
cd Main_System/Pathway_code
python3 run-server-gdrive.py
```

The server will:
- Index documents from your Google Drive folder
- Start HTTP server on `http://127.0.0.1:8745`
- Keep running until you stop it (Ctrl+C or use `./kill_server.sh`)

**Wait for indexing:** The server will index documents from Google Drive. Wait until you see "Server started" message.

### Step 2: Run the Framework

**In a new terminal window:**

```bash
cd Main_System
python3 main.py
```

**Run with options:**
```bash
# Run with specific number of test cases
python3 main.py --max-tests 10

# Run with custom CSV file (local file)
python3 main.py --test-csv path/to/test.csv

# Run with custom retrieval parameters
python3 main.py --k 15 --hybrid-search

# Run with custom server location
python3 main.py --pathway-host 127.0.0.1 --pathway-port 8745
```

### Step 3: Stop the Server (when done)

**Option A: Using the helper script**
```bash
./kill_server.sh
```

**Option B: Manual stop**
```bash
lsof -ti:8745 | xargs kill -9
```

## CSV File Setup

The framework supports CSV files from two sources:

### Option 1: Local File (Default)

Place `test.csv` or `train.csv` in the parent directory:
```bash
KDSH'26/
├── Main_System/
│   └── main.py
├── test.csv          # Place CSV here
└── train.csv
```

Then run:
```bash
python3 main.py --test-csv ../test.csv
```

### Option 2: Google Drive (Recommended)

1. **Upload your CSV file** (`test.csv` or `train.csv`) to Google Drive

2. **Share the CSV file** with your service account email:
   - Find your service account email in `service_account.json.json` (field: `client_email`)
   - Right-click the CSV file in Google Drive → Share
   - Add the service account email with "Viewer" permissions

3. **Get the CSV file ID** from the Google Drive URL:
   - Open the CSV file in Google Drive
   - URL format: `https://drive.google.com/file/d/FILE_ID/view`
   - Extract the `FILE_ID` part (the long string between `/d/` and `/view`)

4. **Run with Google Drive file ID:**
```bash
python3 main.py --test-csv-gdrive-id YOUR_FILE_ID
```

**Example:**
```bash
# If your CSV file ID is: 1a2b3c4d5e6f7g8h9i0j
python3 main.py --test-csv-gdrive-id 1a2b3c4d5e6f7g8h9i0j
```

## Command-Line Options

```
--test-csv PATH              Path to local test CSV file (default: ../test.csv)
--test-csv-gdrive-id ID      Google Drive file ID for CSV file (alternative to --test-csv)
--k N                        Number of chunks to retrieve per question (default: 10)
--max-tests N                Maximum number of tests to run (default: all)
--pathway-host HOST          Pathway server host (default: 127.0.0.1)
--pathway-port PORT          Pathway server port (default: 8745)
--hybrid-search              Enable hybrid search (BM25 + semantic) for better accuracy
```

**Note:** Use either `--test-csv` (local file) or `--test-csv-gdrive-id` (Google Drive), not both.

## CSV File Format

The CSV file should have the following columns:

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

## Output Files

After running, the framework generates:

1. **`results.csv`**: Submission file with format:
   - `Story ID`: Test case ID
   - `Prediction`: 1 (consistent) or 0 (inconsistent)
   - `Rationale`: AI-generated explanation for the verdict

2. **`test_results.json`**: Detailed results with:
   - All generated questions
   - Retrieved chunks for each question
   - Individual evaluation results
   - Final verdict and decision rule

## Architecture

1. **Pathway RAG Server**: Indexes documents from Google Drive, provides semantic search API
2. **Question Generator**: Uses Gemini to generate RAG-friendly questions from backstories
3. **Retrieval**: Queries Pathway server for relevant story chunks
4. **Evaluation**: Uses Gemini to evaluate consistency between backstory claims and retrieved chunks
5. **Decision Rule**: Aggregates evaluations into final verdict (CONSISTENT/INCONSISTENT)

## Server Management

### Starting the Server

```bash
./start_server.sh
```

### Checking Server Status

```bash
curl -X POST http://127.0.0.1:8745/v1/statistics \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Stopping the Server

```bash
./kill_server.sh
```

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

### CSV file not found
- Verify CSV file path is correct
- Check file permissions
- Ensure CSV file is in the correct location

### API errors
- Verify Gemini API key is set correctly in `.env`
- Check API quota/limits
- Ensure service account credentials are valid

## File Structure

```
Main_System/
├── main.py                    # Main framework script
├── evaluator.py               # Consistency evaluation logic
├── config.py                  # Configuration and API key loading
├── questions/
│   └── question_generator.py  # Question generation using Gemini
├── Pathway_code/
│   ├── run-server-gdrive.py   # Server startup script
│   ├── server.py              # Pathway VectorStoreServer
│   ├── custom_parser.py       # Document parser
│   └── hybrid_retriever.py    # Hybrid search (optional)
├── start_server.sh            # Helper script to start server
├── kill_server.sh             # Helper script to kill server
├── requirements.txt           # Python dependencies
├── results.csv                # Output file (generated)
└── README.md                  # This file
```

## Example Workflow

```bash
# Terminal 1: Start server
cd Main_System
./start_server.sh

# Wait for "Server started" message...

# Terminal 2: Run framework
cd Main_System
python3 main.py --max-tests 5

# Check results
cat results.csv

# When done, stop server (Terminal 1)
./kill_server.sh
```

