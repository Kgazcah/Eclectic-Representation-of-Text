import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from encoders.encoder_abstract import TextEncoder
import pickle


class TFIDFEncoder(TextEncoder):
  
  def __init__(self,corpus=None, target_column='text_processed', tokenizer=None, ngram_range=(2,4)):
    """
    Parameters
    ----------
    corpus : pandas dataframe
        Contains data stored in a DataFrame
        
    target_column : str, default='text_processed'
        Target column to encode

    tokenizer : callable, default=None
        Override the string tokenization step 
    
    ngram_range : tuple, default=(1,1)
        The lower and upper boundary of the range of n-values for different n-grams to be extracted
    """
    self.corpus=corpus
    self.target_column=target_column
    self.vectorizer=None
    self.terms=None
    self.embeddings=None

    if(tokenizer==None):
      self.vectorizer=TfidfVectorizer(tokenizer=self.token,lowercase=False,ngram_range=ngram_range)
    else:
      self.vectorizer=TfidfVectorizer(tokenizer=tokenizer,lowercase=False,ngram_range=ngram_range)


  def token(self, text):
    return text

  def fit(self):
    self.embeddings = self.vectorizer.fit_transform(self.corpus[self.target_column])
    self.embeddings = self.embeddings.todense()
    self.terms = self.vectorizer.get_feature_names_out()
    return 

  def encode(self, data, load_vectorizer_from='assets/tfidfEncoder.pkl', load_embeddings_from='assets/embeddings_tfidfEncoder.pkl'):
    self.embeddings = pickle.load(open(load_embeddings_from,"rb"))
    self.vectorizer = pickle.load(open(load_vectorizer_from,"rb"))
    self.terms = self.vectorizer.get_feature_names_out()
    avg = self.embeddings.mean(axis=0)
    vectors = []
    new_vector = np.zeros((len(data), len(self.terms)))
    a = set()
    b = set(self.terms)
    for i, phrase in enumerate(data):
      a = set(phrase)
      matches = a&b
      for j, vocabulary in enumerate(self.terms):
        if vocabulary in matches:
          new_vector[i][j] = avg[j]
      vectors.append(new_vector[i])
    new_vector = pd.DataFrame(vectors, columns=self.terms)
    return new_vector

  def save(self, save_embeddings_as='assets/embeddings_tfidfEncoder.pkl',save_vectorizer_as='assets/tfidfEncoder.pkl'):
    self.embeddings = pd.DataFrame(self.embeddings, columns=self.terms)
    pickle.dump(self.embeddings, open(save_embeddings_as, "wb"))
    pickle.dump(self.vectorizer, open(save_vectorizer_as, 'wb'))
    # Deallocation objects
    self.embeddings=None
    self.vectorizer=None
    del self.embeddings

  def getEmbeddings(self, load_embeddings_from='assets/embeddings_tfidfEncoder.pkl'):
    self.embeddings = pickle.load(open(load_embeddings_from,"rb"))
    return self.embeddings