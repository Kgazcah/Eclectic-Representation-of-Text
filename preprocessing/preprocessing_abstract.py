from abc import ABC, abstractmethod

#PreprocessorBuilder
class PreprocessorBuilder(ABC):
  @abstractmethod
  def buildTokenizer(self):
    pass
  @abstractmethod
  def buildCleaner(self):
    pass
  @abstractmethod
  def buildStemmer(self):
    pass
  @abstractmethod
  def buildLemmatizer(self):
    pass