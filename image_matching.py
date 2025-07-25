import requests
from PIL import Image
from io import BytesIO
import mygrad as mg
im



def create_image_database(model, validation_set):
    # Function that maps image IDs to their embeddings
    # We pass in a trained model and the  and the output is a dictionary mapping image IDs to their embeddings
    image_database = {}
    all_image_ids = list(coco_data.get_image_ids())

    all_features = coco_data.image_id_to_embedding(all_image_ids)
    
    

def query_top_k_images(caption_embedding, image_embeddings,k ):
    #returns the top k image ids for a given caption embedding and annd image embedding database
    #use cos similarity
    top_images = []
    
    for 


def display_image(image_url, figsize=(12, 8), )
    # Function that displays an image from the database given an image url
    