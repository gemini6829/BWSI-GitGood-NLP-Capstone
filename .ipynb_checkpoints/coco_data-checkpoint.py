from cogworks_data.language import get_data_path
from pathlib import Path
import numpy as np
from collections import defaultdict
from caption_embedding import tokenize, create_embedding, calculate_IDF

class Coco_Data:

    def __init__(self, coco_data, resnet18_features):
        # filename = get_data_path("captions_train2014.json")
        # with Path(filename).open() as f:
        #     self.coco_data = json.load(f)
        
        # with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
        #     self.resnet18_features = pickle.load(f)
        
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
            self.caption_id_to_captions[caption_id] = cap["caption"]
            
        self.caption_id_to_embedding = {}
        # self.image_id_to_cap_id = defaultdict(list)
        # for cap in coco_data["annotations"]:
        #     self.image_id_to_cap_id[cap["image_id"].append(cap["id"])]
        
        # self.caption_id_to_image_id = defaultdict(list)
        # for cap in coco_data["annotations"]:
        #     self.caption_id_to_image_id[cap["id"].append(cap["image_id"])]
        
        # self.caption_id_to_captions = {
        #     cap["id"] : cap["caption"] for cap in coco_data["annotations"]
        # }

        # self.caption_id_to_embedding = {
        #     caption_id: self.resnet18_features[self.caption_id_to_image_id[caption_id]] for caption_id in self.caption_ids

    def caption_id_to_embedding(self, caption_ids, glove):
        tokenized_captions = [tokenize(caption) for caption in self.caption]

        idf = calculate_IDF(tokenized_captions)
        for tokens in tokenized_captions:
            embedding = create_embedding(tokens, glove, idf)
            self.caption_id_to_embedding = {cap_id : embedding for cap_id in caption_ids}
        return idf

    def image_id_to_embedding(self, image_id):
        vectors = np.zeros((len(image_id), 512))
        for n, id in enumerate(image_id):
            vectors[n] = self.resnet18_features[id]
        return vectors

    def get_url_for_image(self, image_id):
        return self.image_id_to_urls[image_id]
    
    def get_captions_for_image(self, image_id):
        return self.image_id_to_cap_id[image_id]
    
    def get_image_for_caption(self, caption_id):
        return self.caption_id_to_image_id[caption_id]
    
    def get_text_for_caption(self, caption_id):
        return self.caption_id_to_captions[caption_id]
    
    def get_image_ids(self):
        return self.image_ids



    