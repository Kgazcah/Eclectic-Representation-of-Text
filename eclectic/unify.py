import pickle
import pandas as pd

class Unify():
  def __init__(self, embeddings: list):
    """
    Parameters
    ----------
    embeddings : list
        Contains the DataFrame embeddings in the form of a list
    """
    self.embeddings=embeddings
    self.eclecticEmbeddings=None
  

  def flattened(self):
    self.eclecticEmbeddings = pd.concat(self.embeddings, axis=1)
    print(self.eclecticEmbeddings.head(n=self.eclecticEmbeddings.shape[0]))
    # self.save()
    
  def sequential(self):
    pass

  def multichannel(self):
    pass

  def save(self, save_as="assets/eclecticEmbeddings.pkl"):
    self.eclecticEmbeddings.to_pickle(save_as) 

  def getEmbeddings(self, load_embeddings_from='assets/eclecticEmbeddings.pkl'):
    self.embeddings = pickle.load(open(load_embeddings_from,"rb"))
    return self.embeddings