import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import math
import pickle
import logging

logging.basicConfig(level=logging.INFO)

class SOMEncoder():
  #Initialization
    def __init__(self, latticeX=5, latticeY=3, learningRatio=0.1, epochs=10):
        """
        Parameters
        ----------
        latticeX : int
            Size of the lattice or grid SOM in axis=0

        latticeY : int
            Size of the lattice or grid SOM in axis=1

        learningRatio : float
            Adaptation parameter of the learning process
            
        epochs : int, optional
            Maximum number of iterations
        """
        self.latticeX=latticeX
        self.latticeY=latticeY
        self.learningRatio=learningRatio
        self.epochs=epochs
        self.embeddings=None
        self.lattice=None

  #Training SOM network    
    def fit(self, data):
      self.X=data.to_numpy()
      self.lattice = sps.somNet(self.latticeX, self.latticeY, self.X, PBC=True)
      self.lattice.train(self.learningRatio, self.epochs)
      i=0
      result_matrix = []
      for row in np.array(self.X):
        result_matrix.append(self.projection(row))  
        s = "{}% documents processed".format(round(i/self.X.shape[0],0)*100)
        logging.debug(s)
        print(s,end=len(s) * '\b')
        i=i+1
      self.embeddings = result_matrix
      


    def encode(self,data, load_lattice_from='assets/somLattice.pkl'):
      self.lattice = pickle.load(open(load_lattice_from,"rb"))
      logging.debug("Type:",type(self.lattice))
      self.X=data.to_numpy()
      i=0
      result_matrix = []
      for row in np.array(self.X):
        result_matrix.append(self.projection(row))  
        s = "{}% documents processed".format(round(i/self.X.shape[0],0)*100)
        print(s,end=len(s) * '\b')
        i=i+1
      self.embeddings = result_matrix
     

    def save(self, save_lattice_as='assets/somLattice.pkl'):
       pickle.dump(self.lattice, open(save_lattice_as, 'wb'))
       #Deallocation objects
       self.embeddings=None
    

    def saveEmbeddings(self, save_embeddings_as='assets/embeddings_somEncoder.pkl'):
      self.embeddings=pd.DataFrame(self.embeddings)
      pickle.dump(self.embeddings, open(save_embeddings_as, "wb"))


    def projection(self, x):
      x=x.reshape(1,-1)
      image_document=np.zeros(shape=self.latticeX*self.latticeY, dtype=np.float32)
      for j,node in enumerate(self.lattice.nodeList):
          dist=distance.euclidean(x, node.weights)
          image_document[j]=1.0-math.exp(-1.0/dist)
      image_document=image_document.reshape(self.latticeX*self.latticeY)
      return image_document
      