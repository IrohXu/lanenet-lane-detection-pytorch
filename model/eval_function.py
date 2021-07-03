import torch
from torch.autograd import Variable
import numpy as np

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class Eval_Score():
    # IoU and F1(Dice)

    def __init__(self, y_pred, y_true, threshold = 0.5):
        input_flatten = np.int32(y_pred.flatten() > threshold)
        target_flatten = np.int32(y_true.flatten() > threshold)
        self.intersection = np.sum(input_flatten * target_flatten)
        self.sum = np.sum(target_flatten) + np.sum(input_flatten)
        self.union = self.sum - self.intersection
    
    def Dice(self, eps=1):
        return np.clip(((2. * self.intersection) / (self.sum + eps)), 1e-5, 0.99999)
    
    def IoU(self):
        return self.intersection / self.union
