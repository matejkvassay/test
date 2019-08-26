import torch
from torch.nn.modules.sparse import EmbeddingBag
from torch.autograd import Variable
from fasttext import load_model
import numpy as np

class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, words):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = Variable(torch.LongTensor(word_subinds))
        offsets = Variable(torch.LongTensor(word_offsets))
        return super().forward(ind, offsets).data.numpy()