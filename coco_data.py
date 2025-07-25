from cogworks_data.language import get_data_path
from pathlib import Path
import numpy as np
from collections import defaultdict
from caption_embedding import tokenize, create_embedding, calculate_IDF

class Coco_Data:

    def __init__(self, coco_data, resnet18_features, glove):

        self.coco_data = coco_data
        self.resnet18_features = resnet18_features

        self.images = [img for img in self.coco_data["images"] if img["id"] in self.resnet18_features]
        self.image_ids = set(img["id"] for img in self.images)

        self.captions = [cap for cap in coco_data["annotations"] if cap["image_id"] in self.resnet18_features]
        self.caption_ids = set(cap["id"] for cap in self.captions)
        
        self.image_id_to_urls = {
            image["id"] : image["coco_url"] for image in self.images
        }

        self.image_id_to_cap_id = defaultdict(list)
        self.caption_id_to_image_id = {}
        self.caption_id_to_captions = {}

        for cap in self.captions:
            image_id = cap["image_id"]
            caption_id = cap["id"]
            self.image_id_to_cap_id[image_id].append(caption_id)
            self.caption_id_to_image_id[caption_id] = image_id
            self.caption_id_to_image_id[caption_id] = image_id
            self.caption_id_to_captions[caption_id] = cap["caption"]
            
        self.caption_id_to_embedding = {}
        tokenized_captions = [tokenize(self.text_for_caption(caption)) for caption in self.caption_ids]
        idf = calculate_IDF(tokenized_captions)
        
        for caption_id, tokens in zip(self.caption_ids, tokenized_captions):
            embedding = create_embedding(tokens, idf, glove)
            self.caption_id_to_embedding[caption_id] = embedding

    def embedding_for_caption_id(self, caption_id):
        return self.caption_id_to_embedding[caption_id]

    def embedding_for_image_id(self, image_id):
        vectors = np.zeros((len(image_id), 512))
        for n, id in enumerate(image_id):
            vectors[n] = self.resnet18_features[id]
        return vectors

    def url_for_image(self, image_id):
        return self.image_id_to_urls[image_id]
    
    def captions_for_image(self, image_id):
        return self.image_id_to_cap_id[image_id]
    
    def image_for_caption(self, caption_id):
        return self.caption_id_to_image_id[caption_id]
    
    def text_for_caption(self, caption_id):
        return self.caption_id_to_captions[caption_id]
    
    def get_image_ids(self):
        return self.image_ids

    def get_coco_data(self):
        return self.coco_data
    
    def get_captions(self):
        return self.captions
    
    def get_reset18_features(self):
        return self.resnet18_features



