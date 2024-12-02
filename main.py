from flask import Flask, render_template, request, redirect, url_for,stream_with_context
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import torch
import os.path as path
import requests
from concurrent.futures import ThreadPoolExecutor
from helper import FacePreprocessor
from helper import list_album_items, download_media_item
from VariationalAutoencoder import VariationalAutoencoder
import io
from flask import Response, request
from contextlib import redirect_stdout
from model_helper import load_images, get_images_base64
import pytorch_lightning as PL


app = Flask(__name__)

# Google Photos API setup
SCOPES = [
    'https://www.googleapis.com/auth/photoslibrary',
    'https://www.googleapis.com/auth/photoslibrary.readonly'
]
creds_path = 'client_secret.json'

def get_google_photos_service():
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=5432)
    return build('photoslibrary', 'v1', credentials=creds, static_discovery=False)


# Home route to list albums
# @app.route('/')
# def list_albums():

#     return redirect(url_for('albums'))

@app.route('/')
def home():
    global service
    service = get_google_photos_service()
    return render_template('home.html')


@app.route('/albums',methods=['GET', 'POST'])
def list_albums():
    results = service.albums().list().execute()
    albums = results.get('albums', [])
    return render_template('albums.html', albums=albums)

# Function to list items in an album with pagination
def list_album_items(service, album_id):
    media_items = []
    next_page_token = None

    while True:
        body = {'albumId': album_id}
        if next_page_token:
            body['pageToken'] = next_page_token

        results = service.mediaItems().search(body=body).execute()
        media_items.extend(results.get('mediaItems', []))
        next_page_token = results.get('nextPageToken')

        if not next_page_token:
            break

    return media_items


#select and download
@app.route('/select_album/<album_id>', methods=['GET', 'POST'])
def select_album(album_id):
    folder_path = f'downloaded_photos_{album_id}'
    output_folder = os.path.join(folder_path, 'faces')

    # Ensure the folder paths exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    if request.method == 'GET':
        # List album items
        media_items = list_album_items(service, album_id)

        if not media_items:
            return f"No media items found in album with ID: {album_id}"

        # Download media items
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(download_media_item, item, folder_path) for item in media_items]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading item: {e}")

        # Preprocess faces
        preprocessor = FacePreprocessor(input_folder=folder_path, output_folder=output_folder)
        preprocessor.preprocess_faces()

        # Redirect to train_model with the preprocessed folder
        return redirect(url_for('train_model', folder_path=output_folder))

        # Redirect to the training step
    #return redirect(url_for('preprocess_faces', input_folder=folder_path))

    # Render the confirmation template
    #return render_template('confirm_download.html', album_id=album_id)

# Function to download a media item
def download_media_item(media_item, folder_path):
    base_url = media_item['baseUrl']
    filename = media_item['filename']
    download_url = f'{base_url}=d'

    media_res = requests.get(download_url)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as file:
        file.write(media_res.content)

    print(f'Downloaded {filename}')

# @app.route('/preprocess_faces', methods=['GET', 'POST'])
# def preprocess_faces():
#     if request.method == 'POST':
#         # Input and output directories
#         input_folder = request.form['input_folder'] # folders possilbe injections in future
#         output_folder=path.join(input_folder,'faces')
#         os.mkdir(output_folder)
#         # Preprocess faces using the helper class
#         preprocessor = FacePreprocessor(input_folder=input_folder, output_folder=output_folder)
#         preprocessor.preprocess_faces()

#         # Redirect to training step
#         return redirect(url_for('train_model', folder_path=output_folder))

#     # Render the form for face preprocessing
#     return render_template('preprocess_faces.html')
@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    folder_path = request.args.get('folder_path')

    if not folder_path or not os.path.exists(folder_path):
        return "No valid folder found for training."

    def generate_logs():
        try:
            # Load your dataset (train_loader setup assumed)
            dataset = load_images(folder_path)  # Replace with your data loading logic
            train_loader = torch.utils.data.DataLoader(
                dataset, shuffle=True, batch_size=64, num_workers=3, persistent_workers=True
            )

            # Initialize model and trainer
            model = VariationalAutoencoder()
            trainer = PL.Trainer(accelerator='gpu', max_epochs=200, log_every_n_steps=1)

            # Capture training logs
            with io.StringIO() as buf, redirect_stdout(buf):
                trainer.fit(model, train_loader)
                for line in buf.getvalue().splitlines():
                    yield f"data: {line}\n\n"  # SSE format
                    buf.seek(0)
                    buf.truncate(0)
            trainer.save_checkpoint("vae_model.ckpt")
        except Exception as e:
            yield f"data: Error during training: {str(e)}\n\n"

    # Use stream_with_context to ensure logs are streamed correctly
    return Response(stream_with_context(generate_logs()), content_type='text/event-stream')

@app.route('/generate_images')
def generate_images():
    checkpoint_path = "vae_model.ckpt"
    vae = VariationalAutoencoder.load_from_checkpoint(checkpoint_path)

    z = torch.randn(64, 64)
    y_pred = vae.decode(z)

    base64_images = get_images_base64(y_pred)

    return render_template('view_images_base64.html', images=base64_images)

if __name__ == '__main__':
    app.run(debug=True)
