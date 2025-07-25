import numpy as np
from collections import defaultdict
import re, string

# re's punctiation identifier 
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def tokenize(caption):
    
    caption = caption.lower()
    cleaned = punc_regex.sub(" ", caption)
    tokens = re.split(r'\s+', cleaned)

    return tokens


#takes in 2d np array of tokenized captions
def calculate_IDF(documents: np.ndarray):
    num_caps = len(documents)
    idf_scores = {}

    #calculate the num of occurences of each word in the captions
    freq_doc = defaultdict(int)
    for caption in documents:
        # this way we only add 1 to freq_doc even if there's multiple of a word in a caption
        unique_words = set(caption) 
        for word in unique_words:
            freq_doc[word] += 1
    
    #calc idf
    for word in freq_doc:
        idf_scores[word] = np.log10(num_caps / freq_doc[word])

    return idf_scores



#takes in list of caption tokens, dictionary of IDFs and loaded GloVe-200 embeddings
def create_embedding(tokens, idfs, glove):
    embeds = np.array([glove[token] if token in glove else np.zeros(200) for token in tokens])
    weights = np.array([idfs[token] if token in idfs else 0.0 for token in tokens])

    weighted_sum = np.sum(embeds * weights.reshape(-1, 1), axis=0)

    length = np.linalg.norm(weighted_sum)
    if length != 0:
        norm = weighted_sum / length
    else:
        norm = weighted_sum

    return norm