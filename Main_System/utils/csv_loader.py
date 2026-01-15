"""
CSV Loader with Google Drive Support

Supports loading CSV files from:
1. Local filesystem
2. Google Drive folder (searches for CSV file in folder)
3. Google Drive file ID (direct file access)
"""

import os
import csv
import io
from pathlib import Path
from typing import List, Dict


def download_csv_from_gdrive(file_id: str, credentials_path: str = None) -> List[Dict[str, str]]:
    """
    Download a CSV file from Google Drive using service account credentials.
    
    Args:
        file_id: Google Drive file ID
        credentials_path: Path to service account JSON file (default: service_account.json.json)
        
    Returns:
        List of dictionaries containing CSV rows
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError
    except ImportError:
        raise ImportError(
            "Google API client libraries not installed. "
            "Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        )
    
    # Determine credentials path
    if credentials_path is None:
        credentials_path = Path(__file__).parent / "service_account.json.json"
    
    credentials_path = Path(credentials_path)
    if not credentials_path.exists():
        raise FileNotFoundError(f"Service account credentials not found at {credentials_path}")
    
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    # Build Drive API client
    service = build('drive', 'v3', credentials=credentials)
    
    try:
        # Download file
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file_content.seek(0)
        
        # Parse CSV
        csv_content = file_content.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)
        
        return rows
    
    except HttpError as error:
        raise Exception(f"Error downloading file from Google Drive: {error}")


def find_csv_in_gdrive_folder(folder_id: str, csv_filename: str = "test.csv", credentials_path: str = None) -> str:
    """
    Find a CSV file in a Google Drive folder and return its file ID.
    
    Args:
        folder_id: Google Drive folder ID
        csv_filename: Name of CSV file to find (default: "test.csv")
        credentials_path: Path to service account JSON file
        
    Returns:
        File ID of the CSV file
        
    Raises:
        FileNotFoundError: If CSV file not found in folder
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        raise ImportError(
            "Google API client libraries not installed. "
            "Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        )
    
    if credentials_path is None:
        credentials_path = Path(__file__).parent / "service_account.json.json"
    
    credentials_path = Path(credentials_path)
    if not credentials_path.exists():
        raise FileNotFoundError(f"Service account credentials not found at {credentials_path}")
    
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    service = build('drive', 'v3', credentials=credentials)
    
    try:
        # Search for CSV file in folder
        query = f"'{folder_id}' in parents and name='{csv_filename}'"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=10
        ).execute()
        
        files = results.get('files', [])
        for file in files:
            if file['name'] == csv_filename:
                return file['id']
        
        raise FileNotFoundError(f"CSV file '{csv_filename}' not found in Google Drive folder {folder_id}")
    
    except HttpError as error:
        raise Exception(f"Error searching folder: {error}")


def load_csv_file(file_path: str = None, gdrive_file_id: str = None, gdrive_folder_id: str = None, 
                  csv_filename: str = "test.csv", credentials_path: str = None) -> List[Dict[str, str]]:
    """
    Load CSV file from local filesystem or Google Drive.
    
    Args:
        file_path: Path to local CSV file (if loading from filesystem)
        gdrive_file_id: Google Drive file ID (if loading specific file from Google Drive)
        gdrive_folder_id: Google Drive folder ID (will search for csv_filename in this folder)
        csv_filename: Name of CSV file to find in folder (default: "test.csv")
        credentials_path: Path to service account JSON file (required for Google Drive)
        
    Returns:
        List of dictionaries containing CSV rows
        
    Raises:
        ValueError: If neither file_path, gdrive_file_id, nor gdrive_folder_id is provided
        FileNotFoundError: If local file doesn't exist or CSV not found in folder
        Exception: If Google Drive download fails
    """
    if gdrive_folder_id:
        # Find and load CSV from Google Drive folder
        print(f"  📥 Searching for '{csv_filename}' in Google Drive folder...", flush=True)
        file_id = find_csv_in_gdrive_folder(gdrive_folder_id, csv_filename, credentials_path)
        print(f"  ✓ Found {csv_filename} (ID: {file_id})", flush=True)
        return download_csv_from_gdrive(file_id, credentials_path)
    
    elif gdrive_file_id:
        # Load from Google Drive using direct file ID
        print(f"  📥 Loading CSV from Google Drive (File ID: {gdrive_file_id})...", flush=True)
        return download_csv_from_gdrive(gdrive_file_id, credentials_path)
    
    elif file_path:
        # Load from local filesystem
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        print(f"  📂 Loading CSV from local file: {file_path}...", flush=True)
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            return list(csv_reader)
    
    else:
        raise ValueError("Either file_path, gdrive_file_id, or gdrive_folder_id must be provided")
