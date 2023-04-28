from sklearn.naive_bayes  import GaussianNB 
from classifier_abstract import Classifier


class NaiveBayes(Classifier):
	def __init__(self, priors=None, var_smoothing=1e-09):
		"""
		Parameters
		---------------
		priors : array-like of shape (n_classes,), default=None
			Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.

		var_smoothingfloat, default=1e-9
			Portion of the largest variance of all features that is added to variances for calculation stability.
		"""
		self.classifier=GaussianNB(priors=priors, var_smoothing=var_smoothing)
	
	def fit(self, X_train, y_train):
		super().fit(X_train, y_train)
		
	def predict(self, X_test):
		y_pred = super().predict(X_test)
		return y_pred


	def metrics(self, y_test, y_pred):
		super().metrics(y_test, y_pred)

