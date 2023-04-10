from preprocessing.genericpreprocessor import GenericPreprocessorBuilder
from preprocessing.specializedpreprocessor import SpecializedPreprocessorBuilder
from preprocessing.director import Director
from encoders.tfidf import TFIDFEncoder 
from encoders.doc2vec import Doc2VecEncoder
from encoders.bert import BertEncoder
from encoders.som import SOMEncoder
import pandas as pd
import numpy as np


dataset = pd.read_csv('dataset.csv', encoding="utf-8")
target_column = 'review_body'
corpus = dataset[target_column].to_numpy()

# PREPROCESSING WITH SPACY
# builder_spacy=SpecializedPreprocessorBuilder(corpus)
# spacy_classic_preproc_corpus=director.makeClassicPreprocessor(builder_spacy)
# print('using SPACY')
# spacy_classic_preproc_corpus['sentiment']=dataset.sentiment.to_numpy()
# print(spacy_classic_preproc_corpus.head(n=10))
# print("El tipo de dato de spacy es", type(spacy_classic_preproc_corpus))

#CORPUS PRUEBA
# corpus=[["Ellas también tiene que respetarse, se visten así y luego se quejan cuando les pasa algo"],
# ["Si sigues con esa actitud, voy a violarte"],
# ["Hice todo lo que pude por querer tenerte pero fracase, por eso te mataré"],
# ["Vete a la cocina, traeme unos sanduches"],
# ["Me gusta estar con ella porque sabe cocinar"],
# ["Y si son escoria no es culpa mía"],
# ["Jajajajajajjajajajajajajjaa ay Dios la mujer es maldad me voy de aqui, que puta pavera"],
# ["Eres la mujer de mi vida"],
# ["Eres la mujer de mi vida, pero eres una pendeja"]]

director=Director()


# PREPROCESSING FOR TFIDF and DOC2VEC
builder_nltk=GenericPreprocessorBuilder(corpus)
nltk_classic_preproc=director.makeClassicPreprocessor(builder_nltk)
print(nltk_classic_preproc.head)
print(f'{type(nltk_classic_preproc)}')


"""
#PREPROCESSING FOR BERT
builder_nltk_bert = GenericPreprocessorBuilder(corpus)
nltk_classic_preproc_bert = director.makeBasicPreprocessor(builder_nltk_bert)
print(nltk_classic_preproc_bert.head())
"""

#ENCODING WITH TFIDF
"""
miObj=TFIDFEncoder(nltk_classic_preproc,target_column='text_processed')
miObj.fit()
miObj.save()
tfidf_res = miObj.encode([["no","sirve","calienta","mucho","computado"]])
print(tfidf_res)
tfidfEmbeddings=miObj.getEmbeddings()
# print(tfidfEmbeddings)
"""

# ENCODING WITH DOC2VEC
miObj2=Doc2VecEncoder(nltk_classic_preproc, target_column = 'text_processed', vector_size=100)
miObj2.fit()
miObj2.save()
doc2vec_res = miObj2.encode([["no","sirve","calienta","mucho","computado"]])
print(doc2vec_res)
doc2vecEmbeddings=miObj2.getEmbeddings()
# print(doc2vecEmbeddings)



"""
# ENCODING WITH BERT
miObj3 = BertEncoder(nltk_classic_preproc_bert, target_column = 'text_processed')
miObj3.fit()
miObj3.save()
bert_res = miObj3.encode([["No sirve la computadora se calienta mucho"]])
print(bert_res)
bertEmbeddings=miObj3.getEmbeddings()
# print(bertEmbeddings)
"""
exit()

#Training SOM for TFIDF
somTFIDF=SOMEncoder(latticeX=5,latticeY=5)
somTFIDF.fit(tfidfEmbeddings)
somTFIDF.save(save_lattice_as='assets/somTFIDFLattice.pkl')
# somEmbeddings=som.getEmbeddings()
# print(somEmbeddings)
resultTFIDF=somTFIDF.encode(tfidf_res, load_lattice_from='assets/somTFIDFLattice.pkl')
print(f'{resultTFIDF =}')

#Training SOM for DOC2VEC
somDOC2VEC=SOMEncoder(latticeX=5,latticeY=5)
somDOC2VEC.fit(doc2vecEmbeddings)
somDOC2VEC.save(save_lattice_as='assets/somDOC2VECLattice.pkl')
resultDOC2VEC=somDOC2VEC.encode(doc2vec_res,load_lattice_from='assets/somDOC2VECLattice.pkl')
print(f'{resultDOC2VEC =}')


#Training SOM for BERT
somBERT=SOMEncoder(latticeX=5,latticeY=5)
somBERT.fit(bertEmbeddings)
somBERT.save(save_lattice_as='assets/somBERTLattice.pkl')
resultBERT=somBERT.encode(bert_res,load_lattice_from='assets/somBERTLattice.pkl')
print(f'{resultBERT =}')
