import tensorflow.keras as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as  sklearn_shuffle
from imblearn.under_sampling import RandomUnderSampler
import math
import scipy

class UnderSampler(K.utils.Sequence):

    def __init__(self, X, y, batch_size):
        self.rus=RandomUnderSampler(sampling_strategy='not minority')
        self.X, self.y = X, y
        self.batch_size = batch_size
        #len
        self._shuffle()
        self.length= math.ceil(self.X_u.shape[0] / self.batch_size)

    def _shuffle(self):
        self.X_u, self.y_u=self.rus.fit_resample(self.X,self.y)
        self.X_u,self.y_u = sklearn_shuffle(self.X_u,self.y_u)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_x = self.X_u[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_u[idx * self.batch_size:(idx + 1) * self.batch_size]
        if scipy.sparse.issparse(batch_x):
            batch_x=batch_x.todense()
        return batch_x, batch_y
    
    def on_epoch_end(self):
        self._shuffle()
        
class RandomSampler(K.utils.Sequence):

    def __init__(self, X, y, batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.sampling_size=np.min([x for (_,x) in dict(Counter(y)).items()])*len(set(y))
        #len
        self._shuffle()
        self.length= math.ceil(self.X_u.shape[0] / self.batch_size)

    def _shuffle(self):
        self.X_u,_,self.y_u,_=train_test_split(self.X,self.y,shuffle=True,train_size=self.sampling_size)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_x = self.X_u[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_u[idx * self.batch_size:(idx + 1) * self.batch_size]
        if scipy.sparse.issparse(batch_x):
            batch_x=batch_x.todense()
        return batch_x, batch_y
    
    def on_epoch_end(self):
        self._shuffle()