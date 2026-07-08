import os
import sys
import base64
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate_gmail():
    """Authenticates the user using credentials.json and saves/restores token.json."""
    creds = None
    token_path = Path(os.environ.get("TOKEN_PATH", "token.json"))
    creds_path = Path(os.environ.get("CREDENTIALS_PATH", "credentials.json"))

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                creds = None

        if not creds:
            if not creds_path.exists():
                print("\n" + "="*80)
                print("MISSING credentials.json!")
                print("="*80)
                print("To sync your real Gmail inbox, you need to create a Google Cloud Project and download credentials.json:")
                print("1. Go to: https://console.cloud.google.com/")
                print("2. Create a new project, and search for 'Gmail API'. Click Enable.")
                print("3. Go to 'OAuth consent screen', configure it as an External user, and add your email as a test user.")
                print("4. Go to 'Credentials' -> 'Create Credentials' -> 'OAuth client ID'. Select 'Desktop App'.")
                print("5. Click Download JSON, rename it to 'credentials.json', and place it in:")
                print(f"   {creds_path.absolute()}")
                print("="*80 + "\n")
                raise FileNotFoundError("credentials.json is missing. Please follow the instructions above.")
            
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            
    return creds

def fetch_and_save_emails(max_results=50, output_dir="gmail_emails"):
    """Fetches real emails using the Gmail API and saves them as raw .eml files."""
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to Gmail API...")
    try:
        # List messages from inbox
        results = service.users().messages().list(userId="me", maxResults=max_results, q="category:primary").execute()
        messages = results.get("messages", [])

        if not messages:
            logger.info("No messages found in Inbox.")
            return

        logger.info(f"Found {len(messages)} messages. Downloading raw emails as .eml...")
        
        for i, msg in enumerate(messages):
            msg_id = msg["id"]
            
            # Fetch message in RAW format
            message_data = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
            
            # Decode raw message bytes
            raw_content = base64.urlsafe_b64decode(message_data["raw"].encode("ASCII"))
            
            # Create a clean filename
            filename = f"email_{i+1:05d}_{msg_id}.eml"
            filepath = out_path / filename
            
            with open(filepath, "wb") as f:
                f.write(raw_content)
                
            if (i + 1) % 10 == 0 or i + 1 == len(messages):
                logger.info(f"Downloaded {i + 1}/{len(messages)} emails...")
                
        logger.info(f"Successfully saved {len(messages)} emails to {out_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Error fetching emails from Gmail API: {e}")
        raise e

if __name__ == "__main__":
    max_emails = 50
    if len(sys.argv) > 1:
        try:
            max_emails = int(sys.argv[1])
        except ValueError:
            pass
            
    try:
        fetch_and_save_emails(max_results=max_emails)
    except FileNotFoundError:
        sys.exit(1)
