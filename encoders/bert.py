import pandas as pd
import numpy as np
import transformers as ppb
import torch
from encoders.encoder_abstract import TextEncoder
import pickle

class BertEncoder(TextEncoder):
  def __init__(self, corpus, target_column='text_processed'):
    """
    Parameters
    ----------
    corpus : pandas dataframe
        Contains data stored in a DataFrame
        
    target_column : str, default='text_processed'
        Target column to encode
    """
    self.corpus=corpus[:10]
    self.target_column=target_column
    self.vectorizer=None
    self.embeddings=None
    self.tokenizer=None
    self.terms=None
  

  def fit(self):
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-multilingual-cased')
    self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    self.vectorizer = model_class.from_pretrained(pretrained_weights)
    documents = pd.DataFrame(data=self.corpus[self.target_column].to_numpy())
    # tokenize with the tokenizer of BERT
    tokenized = documents[0].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
    # find the largest tokenized sentence
    max_len = 0
    for i in tokenized.values:
      l = len(i)
      if l > max_len:
        max_len = l
    #padding all lists to the same size to process fastest each sentence
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # adding an attention mask to ignore the padding we added, 
    # the tokens will be 1 (0 will be ignored)
    attention_mask = np.where(padded != 0, 1, 0)

    #turning into a tensor instead of an array
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    #Disabling gradient calculation is useful for inference, when you are sure 
    #that you will not call Tensor.backward(). It will reduce memory consumption 
    #for computations that would otherwise have requires_grad=True.
    with torch.no_grad():
      last_hidden_states_pred = self.vectorizer(input_ids,attention_mask=attention_mask)

    #taking out the CLS token added and turning into a dataframe to print
    self.embeddings = last_hidden_states_pred[0][:,0,:].numpy()
    self.terms = np.array(range(self.embeddings.shape[1]))
    self.save()


  def encode(self, data, load_embeddings_from='assets/embeddings_bertEncoder.pkl', load_vectorizer_from='assets/bertEncoder.pkl', load_tokenizer_from='assets/bertTokenizer.pkl'):
    self.vectorizer = pickle.load(open(load_vectorizer_from,"rb"))
    self.tokenizer = pickle.load(open(load_tokenizer_from,"rb"))
    
    data = pd.DataFrame(data)
    tokenized = data[0].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

    #Padding and masking
    #find the largest tokenized sentence
    max_len = 0
    for i in tokenized.values:
      if len(i) > max_len:
        max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)


    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    
    with torch.no_grad():
      last_hidden_states_pred = self.vectorizer(input_ids, attention_mask=attention_mask)

    new_phrase = last_hidden_states_pred[0][:,0,:].numpy()
    new_phrase = pd.DataFrame(new_phrase, columns=self.terms)
    return new_phrase


  def save(self,save_embeddings_as='assets/embeddings_bertEncoder.pkl',save_vectorizer_as='assets/bertEncoder.pkl',save_tokenizer_as='assets/bertTokenizer.pkl' ):
    self.embeddings = pd.DataFrame(self.embeddings, columns=self.terms)
    pickle.dump(self.embeddings, open(save_embeddings_as, "wb"))

    # save model and tokenizer with pickle
    pickle.dump(self.vectorizer, open(save_vectorizer_as, 'wb'))
    pickle.dump(self.tokenizer, open(save_tokenizer_as, 'wb'))

    # Deallocation objects
    self.tokenizer=None
    self.vectorizer=None
    self.embeddings=None
  
  def getEmbeddings(self, load_embeddings_from='assets/embeddings_bertEncoder.pkl'):
    if self.embeddings==None:
      self.embeddings = pickle.load(open(load_embeddings_from,"rb"))
    return self.embeddings
