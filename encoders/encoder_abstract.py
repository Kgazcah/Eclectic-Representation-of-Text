from abc import ABC, abstractmethod


#Create the abstraction of text encoders 
class TextEncoder(ABC):
  @abstractmethod
  def fit(self):
    pass
  @abstractmethod
  def encode(self):
    pass
  @abstractmethod
  def save(self):
    pass
  @abstractmethod
  def getEmbeddings(self):
    pass
