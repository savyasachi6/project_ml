from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Path to the credentials.json file downloaded from Google Cloud Console
creds_path = 'client_secret.json'

SCOPES = [
'https://www.googleapis.com/auth/photoslibrary',
'https://www.googleapis.com/auth/photoslibrary.readonly',
'https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata',
'https://www.googleapis.com/auth/photoslibrary.readonly.originals'
]

# Set up the flow and initiate authentication
flow = InstalledAppFlow.from_client_secrets_file(
    creds_path, SCOPES, redirect_uri='http://localhost:5432/'
)

# This will open the authentication flow in the browser
creds = flow.run_local_server(port=5432)




# Build the API client
service = build('photoslibrary', 'v1', credentials=creds,static_discovery=False)

# List albums from Google Photos
results = service.albums().list().execute()
albums = results.get('albums', [])

if not albums:
    print('No albums found.')
else:
    for album in albums:
        print(f"Album:, Id: {album['title'], album['id']}")