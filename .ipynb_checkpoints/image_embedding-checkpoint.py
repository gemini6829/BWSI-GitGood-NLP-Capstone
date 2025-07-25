import numpy as np
import mygrad as mg
from mygrad import Tensor
from mygrad.nnet.losses import margin_ranking_loss

def loss_accuracy(w_caption, w_true_img, w_confuser_img):
    
    #find dot product for similarity
    a = (w_true_img * w_caption).sum(axis = 1)
    b = (w_confuser_img * w_caption).sum(axis = 1)
    y = mg.tensor(np.ones(a.shape))

    loss = margin_ranking_loss(a, b, y, margin=1.0)
    accuracy = np.mean(a.data > b.data)
    
    return loss, accuracy