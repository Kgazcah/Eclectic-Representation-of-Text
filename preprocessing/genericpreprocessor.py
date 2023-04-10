from preprocessing.preprocessing_abstract import PreprocessorBuilder
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import stopwords
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')
import re



#Base functions
#Funcion para trabajar con expresiones regulares
def remove_matches(text, rx, replace=""):
  clean = re.compile(rx)
  return clean.sub(replace,str(text))


#Preprocessor based on NLTK 
class GenericPreprocessorBuilder(PreprocessorBuilder):

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
  
  def buildTokenizer(self, stopwords=True):
    nltk.download('punkt')
    self.corpus['text_processed'] = self.corpus['text_processed'].apply(word_tokenize) 
    if stopwords:
      nltk.download('stopwords')
      #Dictionaries of stop words
      self.stop_words=set(nltk.corpus.stopwords.words('spanish'))
      stop_words_array=['si','sí','aún','aun','ó','¿','¡','ka','co','n','s','m','aa','aaa','[',']','{','}','(',',','``','”','“','x',]
      #Agregar otras palabras o tokens al diccionario
      for w in stop_words_array:
        self.stop_words.add(w)

      self.corpus['text_processed'] =self.corpus['text_processed'].apply(lambda x: [item for item in x if item not in self.stop_words])

  def buildCleaner(self):
    #Eliminar números y caracteres ruidosos
    re_onlychars="[^A-Za-záéíóúÁÉÍÓÚ\s]"
    #Eliminar URL
    re_url="https?://[A-Za-z0-9./]+"
    #Eliminar risas
    re_laugh1="haha+"
    re_laugh2="[jkl](a+h+[jkl])+"
    #Eliminar #
    re_hashtag="#[A-Za-z0-9]+"
    #Eliminar @
    re_arroba="@[A-Za-z0-9]+"
    self.corpus['text_processed']=self.corpus.text.apply(lambda x: remove_matches(x,re_laugh1,""))
    self.corpus['text_processed']=self.corpus['text_processed'].apply(lambda x: remove_matches(x,re_laugh2,""))
    self.corpus['text_processed']=self.corpus['text_processed'].apply(lambda x: remove_matches(x,re_hashtag,""))
    self.corpus['text_processed']=self.corpus['text_processed'].apply(lambda x: remove_matches(x,re_arroba,""))
    self.corpus['text_processed']=self.corpus['text_processed'].apply(lambda x: remove_matches(x,re_url,""))
    self.corpus['text_processed']=self.corpus['text_processed'].apply(lambda x: remove_matches(x,re_onlychars,""))
    #Lowercase
    self.corpus['text_processed']=self.corpus['text_processed'].str.lower()
    
  def buildStemmer(self):
    self.stemmer = nltk.PorterStemmer()
    self.corpus['text_processed'] = self.corpus['text_processed'].apply(lambda x: [self.stemmer.stem(word) for word in x])

  def buildLemmatizer(self):
    self.lemma = nltk.WordNetLemmatizer()
    self.corpus['text_processed'] = self.corpus['text_processed'].apply(lambda x: [self.lemma.lemmatize(word) for word in x])

  def getProduct(self): 
    return self.corpus