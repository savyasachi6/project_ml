import os
import requests
from concurrent.futures import ThreadPoolExecutor

import cv2
import os
import mediapipe as mp
import gc

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

# Function to download a media item
def download_media_item(media_item, folder_path):
    base_url = media_item['baseUrl']
    filename = media_item['filename']
    
    # Download the media item
    download_url = f'{base_url}=d'
    media_res = requests.get(download_url)
    
    # Save the media item to a file
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as file:
        file.write(media_res.content)


class FacePreprocessor:
    def __init__(self, input_folder, output_folder, min_detection_confidence=0.75):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_detection_confidence = min_detection_confidence

        # Mediapipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_faces(self):
        # Initialize Mediapipe face detection
        with self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_detection_confidence) as face_detection:

            for filename in os.listdir(self.input_folder):
                file_path = os.path.join(self.input_folder, filename)

                # Read the image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Could not read file {filename}. Skipping...")
                    continue

                try:
                    # Convert the image to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Perform face detection
                    results = face_detection.process(rgb_image)

                    if not results.detections:
                        print(f"No faces detected in {filename}. Skipping...")
                        continue

                    # Process each detected face
                    for i, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape

                        # Convert relative bounding box to pixel values
                        x = int(bboxC.xmin * w)
                        y = int(bboxC.ymin * h)
                        w_box = int(bboxC.width * w)
                        h_box = int(bboxC.height * h)

                        # Ensure bounding box is within image dimensions
                        x, y = max(0, x), max(0, y)
                        x2, y2 = min(x + w_box, w), min(y + h_box, h)

                        # Crop the face
                        face = image[y:y2, x:x2]

                        # Save cropped face to the output folder
                        new_filename = f"{os.path.splitext(filename)[0]}_face_{i+1}.jpg"
                        output_path = os.path.join(self.output_folder, new_filename)
                        cv2.imwrite(output_path, face)
                        print(f"Cropped face saved to {output_path}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

                finally:
                    # Release memory
                    del image, rgb_image, results
                    gc.collect()

        #print("Face preprocessing completed.")
