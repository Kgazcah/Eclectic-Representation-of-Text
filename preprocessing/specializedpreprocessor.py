from preprocessing.preprocessing_abstract import PreprocessorBuilder
import pandas as pd
import re
import spacy
import spacy_spanish_lemmatizer
import Stemmer
from datetime import datetime

#Base functions
#Funcion para trabajar con expresiones regulares
def remove_matches(text, rx, replace=""):
  clean = re.compile(rx)
  return clean.sub(replace,str(text))


obj = datetime.now()

#Preprocessor for a particular problem 
class SpecializedPreprocessorBuilder(PreprocessorBuilder):

  def __init__(self, corpus):
    """
    Parameters
    ----------
    corpus : list or numpy array
        Contains data stored 
    """  
    self.corpus=pd.DataFrame(corpus,columns=['text'])
    self.stop_words=None
    self.stemmer=None
    self.lemma=None
    self.string=None

  def buildTokenizer(self, stopwords=True):
    # nltk.download('punkt')
    # self.corpus['text_processed'] = self.corpus['text'].apply(word_tokenize)

    nlp = spacy.load("es_core_news_sm")
    texto = self.corpus['text_processed'].values.tolist() 

    N = len(texto)
    otoken = [[]] * N
    if stopwords:
      for i in range(N):
        otoken[i] = [x.text for x in nlp(texto[i]) if not x.is_stop]
    else:
      for i in range(N):
        otoken[i] = [x.text for x in nlp(texto[i])]

    self.corpus['text_processed'] = otoken
    print("tokenizer", obj)

  
  def buildCleaner(self):
    #Eliminar números y caracteres ruidosos
    re_onlychars="[^A-Za-záéíóúÁÉÍÓÚ\s]"
    self.corpus['text_processed']=self.corpus.text.apply(lambda x: remove_matches(x,re_onlychars,""))
    #Lowercase
    self.corpus['text_processed']=self.corpus['text_processed'].str.lower()
    print("cleaner", obj)

  def buildStemmer(self):
    self.stemmer = Stemmer.Stemmer('spanish')
    self.corpus['text_processed'] = [self.stemmer.stemWords(word) for word in self.corpus['text_processed']]
    print("stemmer", obj)

  def buildLemmatizer(self):
    nlp = spacy.load("es_core_news_sm")
    nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")
    self.lemma = [i for i in self.corpus['text_processed']]
    
    for i,phrase in enumerate(self.lemma):
      self.string = ""
      for token in phrase:
       self.string += str(token) + " "
      self.lemma[i] = [to.lemma_ for to in nlp(self.string.strip())]
    self.corpus['text_processed'] = self.lemma 
    print("lemmatizer", obj)

  def getProduct(self):
      return self.corpus   