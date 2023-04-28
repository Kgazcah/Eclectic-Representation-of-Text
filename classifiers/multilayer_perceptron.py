from sklearn.neural_network import MLPClassifier
from classifier_abstract import Classifier


class MLP(Classifier):
	def __init__(self, max_iter=700):
		"""
		Parameters
		---------------
		max_iter : int, default=100
			Maximum number of iterations taken for the solvers to converge.
		"""
		self.classifier=MLPClassifier(max_iter=max_iter)
	
	def fit(self, X_train, y_train):
		super().fit(X_train, y_train)
		
	def predict(self, X_test):
		y_pred = super().predict(X_test)
		return y_pred


	def metrics(self, y_test, y_pred):
		super().metrics(y_test, y_pred)

