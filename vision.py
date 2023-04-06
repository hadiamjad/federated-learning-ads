# Import the required libraries
import io
import os
import json
from tqdm import tqdm

# Import the Google Cloud client library
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your Google Cloud service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'boxwood-weaver-268215-9336db37edf9.json'

# Instantiates a client
client = vision.ImageAnnotatorClient()

folder_path = 'screenshots/'

# List all files in the folder
files = os.listdir(folder_path)

# dictionary to store the image name and its labels
image_labels = {}

for file in tqdm(files, desc='Processing files'):
    # Load the image file into memory
    with io.open(folder_path + file, 'rb') as image_file:
        content = image_file.read()

    # Create an image instance
    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Convert the response to a dictionary
    image_labels[file] = []
    for labels in response.label_annotations:
        image_labels[file].append([labels.description, labels.score, labels.topicality])

    print(file)

# write the dictionary to a JSON file
with open('my_dict.json', 'w') as file:
    json.dump(image_labels, file)


