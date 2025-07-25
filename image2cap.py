# description vector (shape: (512,)) of image -> linear encoder -> caption unit vector (shape: (200,))

from mynn.layers import dense
from mygrad import Tensor
import mygrad as mg
from mygrad.nnet.losses import margin_ranking_loss
import numpy as np
import pickle
from mynn.optimizers.sgd import SGD


# linear encoder for image to caption embedding
# input_dim: 512 (resnet18 feature size)   
# output_dim: 200
class EmbeddingModel:
    def __init__(self, input_dim, output_dim):
        self.encoder = dense(input_dim, output_dim)
    
    #pass in a tensor 
    def __call__(self, x):
        return self.encoder(x)
    
    @property
    def parameters(self):
        return self.encoder.parameters
    
    def update_parameters(self, new_parameters):
        for new_param, model_param in zip(new_parameters, self.parameters):
            model_param.data[:] = new_param.data

#amanda & kritik
def loss_func(w_caption, w_true_img, w_confuser_img):
    #find dot product for similarity
    a = (w_true_img * w_caption).sum(axis = 1)
    b = (w_confuser_img * w_caption).sum(axis = 1)
    y = mg.tensor(np.ones(a.shape))

    loss = margin_ranking_loss(a, b, y, margin=1.0)
    accuracy = np.mean(a.data > b.data)
    
    return loss, accuracy

#extract caption_ID, image_ID, and confuser_ID from coco_data class and form training and validation sets 
def extract_triples(coco_data):
    '''
    Parameters, a coco_data object

    The confuser could be chosen randomly from the set of images that are not the true image for the caption. Ideally though
    it should be chosen so that it is similar to the true image so that the model learns finer nuances 

    Returns: two lists of tuples, one for training and another for validation  (caption_id, image_id, confuser_id)
    '''
    train_triples = []
    val_triples = []

    np.random.seed(42)  

    for i, caption in enumerate(coco_data.get_captions()):
        caption_id = caption["id"]
        image_id = caption["image_id"]
        available_confusers = list(coco_data.get_image_ids() - {image_id})
        confuser_id = np.random.choice(available_confusers)

        if i < (0.8 * len(coco_data.get_captions())):
            train_triples.append((caption_id, image_id, confuser_id))
        else:
            val_triples.append((caption_id, image_id, confuser_id))

    
    return train_triples, val_triples 

def fast_extract_triples(coco_data):
    """
    Efficiently generates (caption_id, image_id, confuser_id) triples for training and validation.
    Scales well to 400,000+ captions by minimizing object creation and using vectorized operations.
    """
    import numpy as np

    np.random.seed(42)

    captions = coco_data.get_captions()  # List of dicts, 400k items
    all_image_ids = np.array(sorted(coco_data.get_image_ids()))  # Numpy array for fast indexing
    image_id_to_index = {img_id: idx for idx, img_id in enumerate(all_image_ids)}  # Optional

    caption_count = len(captions)
    indices = np.random.permutation(caption_count)
    split_idx = int(0.8 * caption_count)

    train_triples = []
    val_triples = []

    for i in indices:
        caption = captions[i]
        caption_id = caption["id"]
        image_id = caption["image_id"]

        # Sample a confuser index that is not equal to the current image
        image_index = image_id_to_index[image_id]
        confuser_index = np.random.randint(len(all_image_ids) - 1)
        if confuser_index >= image_index:
            confuser_index += 1  # skip the true image
        confuser_id = all_image_ids[confuser_index]

        triple = (caption_id, image_id, confuser_id)

        if i < split_idx:
            train_triples.append(triple)
        else:
            val_triples.append(triple)

        if i%1000 == 0:
            print("Processed", i, "captions")

    return train_triples, val_triples


def store_trained_model_weights(model, file_path = "trained_model_weights.pkl"):
    '''
    Parameters:
    - model: EmbeddingModel object
    - file_path: path to save the model weights

    Saves the model weights to a file using pickle.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(model.parameters, f)

def load_trained_model_weights(model, file_path = "trained_model_weights.pkl"):
    with open(file_path, 'rb') as f:
        model_weights = pickle.load(f)
        model.update_parameters(model_weights)



def model_train(model, train_triples, val_triples, coco_data, glove, epochs=10, batch_size=32, lr=0.001):
    '''
    Parameters:
    - model: EmbeddingModel object
    - train_triples: list of tuples (caption_id, image_id, confuser_id) for training
    - val_triples: list of tuples (caption_id, image_id, confuser_id) for validation
    - coco_data: Coco_Data object to access captions and images
    - epochs: number of training epochs
    - batch_size: size of each training batch

    Returns: trained model
    '''
    np.random.seed(42) 
    optimizer = SGD(model.parameters, learning_rate=lr)

    final_accuracy = 0

    for epoch in range (epochs):

        idxs = np.arange(len(train_triples))
        np.random.shuffle(idxs)

        #print("epoch:", epoch)

        total_accuracy = 0

        for batch_cnt in range (0, len(train_triples)// batch_size):

            batch_indicies = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]
            batch = [train_triples[i] for i in batch_indicies]

            truth_features = coco_data.embedding_for_image_id([t[1] for t in batch])
            confuser_features = coco_data.embedding_for_image_id([t[2] for t in batch])

            truth = model(mg.tensor(truth_features))
            confusers = model(mg.tensor(confuser_features)) 

            caption_embeddings = mg.tensor([coco_data.embedding_for_caption_id(t[0]) for t in batch])
            loss, accuracy = loss_func(caption_embeddings, truth, confusers)

            total_accuracy += accuracy

            #print("loss:", loss.data, "accuracy:", accuracy)

            loss.backward()
            optimizer.step()
        
        total_accuracy /= (len(train_triples) // batch_size)
        print("Total accuracy for epoch", epoch, ":", total_accuracy)
        final_accuracy += total_accuracy

    print("Final training accuracy:", final_accuracy / epochs)

    return model  # Return the trained model