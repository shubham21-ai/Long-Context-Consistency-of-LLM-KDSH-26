"""
Helper script to list CSV files in Google Drive folder and get their file IDs.
"""

import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Folder ID from the user's Google Drive
FOLDER_ID = "18Eqn1Yr9u-GsRv5ypt0OsjQoi54OlMD0"

def list_csv_files_in_folder(folder_id: str, credentials_path: str = None):
    """
    List all CSV files in a Google Drive folder and return their IDs.
    """
    if credentials_path is None:
        credentials_path = Path(__file__).parent / "service_account.json.json"
    
    credentials_path = Path(credentials_path)
    if not credentials_path.exists():
        print(f"✗ Service account file not found: {credentials_path}")
        return
    
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    # Build Drive API client
    service = build('drive', 'v3', credentials=credentials)
    
    try:
        # List files in the folder
        query = f"'{folder_id}' in parents and mimeType='text/csv'"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print(f"✗ No CSV files found in folder {folder_id}")
            print("\nTrying to list all files in folder...")
            # Try listing all files
            query_all = f"'{folder_id}' in parents"
            results_all = service.files().list(
                q=query_all,
                fields="files(id, name, mimeType)",
                pageSize=100
            ).execute()
            all_files = results_all.get('files', [])
            
            print(f"\nAll files in folder ({len(all_files)} total):")
            csv_files = []
            for file in all_files:
                file_type = file.get('mimeType', 'unknown')
                if 'csv' in file_type or file.get('name', '').endswith('.csv'):
                    csv_files.append(file)
                    print(f"  ✓ CSV: {file['name']} (ID: {file['id']})")
                else:
                    print(f"  - {file['name']} ({file_type})")
            
            if csv_files:
                print(f"\n✓ Found {len(csv_files)} CSV file(s):")
                for file in csv_files:
                    print(f"  - {file['name']}: {file['id']}")
            else:
                print("\n✗ No CSV files found")
        else:
            print(f"✓ Found {len(files)} CSV file(s) in folder:\n")
            for file in files:
                print(f"  File: {file['name']}")
                print(f"  ID:   {file['id']}")
                print(f"  Command: python3 main.py --test-csv-gdrive-id {file['id']}")
                print()
    
    except HttpError as error:
        print(f"✗ Error accessing folder: {error}")
        if error.resp.status == 404:
            print("  → Folder not found or not accessible")
        elif error.resp.status == 403:
            print("  → Permission denied. Make sure the folder is shared with the service account.")

if __name__ == "__main__":
    print("=" * 70)
    print("Google Drive CSV File Finder")
    print("=" * 70)
    print(f"\nFolder ID: {FOLDER_ID}")
    print(f"Folder URL: https://drive.google.com/drive/folders/{FOLDER_ID}\n")
    print("Listing CSV files...\n")
    
    list_csv_files_in_folder(FOLDER_ID)

