import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import mygrad as mg
from coco_data import Coco_Data
import io
import numpy as np


def create_image_database(model, validation_set, coco_data):
    # Function that maps image IDs to their embeddings
    # We pass in a trained model and the  and the output is a dictionary mapping image IDs to their embeddings
    image_database = {}
    val_image_ids = set()
    for caption_id, image_id, confuser_id in validation_set:
        val_image_ids.add(image_id)
    
    # Convert the set to list 
    all_image_ids = list(val_image_ids)

    # Get the embeddings for the image ID's 
    all_features = coco_data.image_id_to_embedding(all_image_ids)

    #use the model to get embeddings
    with mg.no_grad():
        embeddings = model(mg.tensor(all_features))
    
    # Store them in a dict 
    for i, image_id in enumerate(all_image_ids):
        image_database[image_id] = embeddings[i].data
    
    return image_database 
    

def query_top_k_images(caption_embedding, image_embeddings,k ):
    #returns the top k image ids for a given caption embedding and annd image embedding database
    top_images = []
    
    for i, v in image_embeddings.items():
        dot = np.dot(caption_embedding, v)
        top_images.append((i,dot))

    for i in range(len(top_images)):
        max_id = i
        for j in range(i+1, len(top_images)):
            if top_images[j][1]> top_images[max_id][1]:
                max_id = j
        temp = top_images[i]
        top_images[i] = top_images[max_id]
        top_images[max_id] = temp
 
    # top_images.sort(key=lambda x: x[1], reverse=True)
    return [image_id for image_id, _ in top_images[:k]]


def display_image(url, max_size=12, title=None):
    # Function that displays an image from the database given an image url
    response = requests.get(url)
    
    image = Image.open(io.BytesIO(response.content))
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        figsize = (max_size, max_size / aspect_ratio)
    else:
        figsize = (max_size * aspect_ratio, max_size)
    #adjusts the figure size based on the aspect ratio of the image

    if title:
        plt.title(title)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
