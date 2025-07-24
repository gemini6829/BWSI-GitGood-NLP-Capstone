from cogworks_data.language import get_data_path
from pathlib import Path
import numpy as np
from collections import defaultdict
from caption_embedding import calculate_IDF, create_embedding
from coco_data import Coco_Data
import pickle
import json

filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)
        
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)

coco = Coco_Data(coco_data, resnet18_features)