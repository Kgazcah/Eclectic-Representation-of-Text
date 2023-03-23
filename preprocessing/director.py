from preprocessing.preprocessing_abstract import PreprocessorBuilder

class Director(object):
  def __init__(self):
    self.builder=None  #This object is an instance of any concrete builder

  def makeClassicPreprocessor(self, builder):
      self.builder=builder
      self.builder.buildCleaner()
      self.builder.buildTokenizer()
      self.builder.buildStemmer()
      # self.builder.buildLemmatizer()
      return self.builder.getProduct()
  
  def makeBasicPreprocessor(self, builder):
      self.builder=builder
      self.builder.buildCleaner()
      #self.builder.buildTokenizer()
      #self.builder.buildStemmer()
      #self.builder.buildLemmatizer()
      return self.builder.getProduct()