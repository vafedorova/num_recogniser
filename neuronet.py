import numpy as np
from scipy.special import expit, softmax
from scipy.stats import entropy
import math


class neuronet:
    
    def __init__(self, in_nodes, hid_nodes, out_nodes, w0, w1):
        k = self.in_nodes = in_nodes
        m = self.hid_nodes = hid_nodes
        n = self.out_nodes = out_nodes
        self.w0 = w0
        self.w1 = w1

    def activation1(self, y):
        return expit(y)

    def activation2(self, y):
        return softmax(y)

    def prediction(self, inputs_list):
        inputs = np.array(inputs_list).reshape((len(inputs_list), 1))
        hid_outputs = self.activation1(self.w0 @ inputs)
        out_outputs = self.activation2(self.w1 @ hid_outputs)
        return out_outputs