import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from encoders.encoder_abstract import TextEncoder
import pickle


class Doc2VecEncoder(TextEncoder):
  def __init__(self, corpus, target_column='text_processed', vector_size=100, window=5, min_count=2, epochs=10):
    """
    Parameters
    ----------
    corpus : pandas dataframe
        Contains data stored in a DataFrame
        
    target_column : str, default='text_processed'
        Target column to encode

    vector_size : int, optional
        Dimensionality of the feature vectors

    window : int, optional
        The maximum distance between the current and predicted word within a sentence

    min_count : int, optional
        Ignores all words with total frequency lower than this
    
    epochs : int, default=10
         Number of iterations (epochs) over the corpus
    """
    self.corpus=corpus
    self.target_column=target_column
    self.vectorizer=None
    self.documents=None
    self.embeddings=None
    self.vector_size=vector_size
    self.vectorizer=Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)


  def fit(self):
    self.documents = [TaggedDocument(doc, [(i)]) for i, doc in enumerate(self.corpus[self.target_column])]
    self.vectorizer.build_vocab(self.documents)
    self.vectorizer.train(self.documents, total_examples=self.vectorizer.corpus_count, epochs=self.vectorizer.epochs)
    self.embeddings = self.vectorizer.dv.vectors 
    self.embeddings = pd.DataFrame(self.embeddings)
    

  def encode(self,data,load_vectorizer_from='assets/doc2vecEncoder.pkl'):
    self.vectorizer = pickle.load(open(load_vectorizer_from,"rb"))
    vectors = []
    new_vector = np.empty((len(data),self.vector_size))
    for i, phrase in enumerate(data):
      new_vector[i] = self.vectorizer.infer_vector(phrase)
      vectors.append(new_vector[i])
    new_vector= pd.DataFrame(vectors)
    return new_vector
    
  def save(self,save_embeddings_as='assets/embeddings_doc2vecEncoder.pkl',save_vectorizer_as='assets/doc2vecEncoder.pkl'):
    self.embeddings=pd.DataFrame(self.embeddings)
    pickle.dump(self.embeddings, open(save_embeddings_as, "wb"))
    pickle.dump(self.vectorizer, open(save_vectorizer_as, 'wb'))

    # Deallocation objects
    del self.embeddings
    self.vectorizer=None

  def getEmbeddings(self, load_embeddings_from='assets/embeddings_doc2vecEncoder.pkl'):
    self.embeddings = pickle.load(open(load_embeddings_from,"rb"))
    return self.embeddings