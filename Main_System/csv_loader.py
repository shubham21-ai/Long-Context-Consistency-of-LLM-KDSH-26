"""
CSV Loader with Google Drive Support

Supports loading CSV files from:
1. Local filesystem
2. Google Drive (using service account)
"""

import os
import csv
import io
from pathlib import Path
from typing import List, Dict
from config import load_gemini_api_key


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


def load_csv_file(file_path: str = None, gdrive_file_id: str = None, credentials_path: str = None) -> List[Dict[str, str]]:
    """
    Load CSV file from local filesystem or Google Drive.
    
    Args:
        file_path: Path to local CSV file (if loading from filesystem)
        gdrive_file_id: Google Drive file ID (if loading from Google Drive)
        credentials_path: Path to service account JSON file (required for Google Drive)
        
    Returns:
        List of dictionaries containing CSV rows
        
    Raises:
        ValueError: If neither file_path nor gdrive_file_id is provided
        FileNotFoundError: If local file doesn't exist
        Exception: If Google Drive download fails
    """
    if gdrive_file_id:
        # Load from Google Drive
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
        raise ValueError("Either file_path or gdrive_file_id must be provided")

