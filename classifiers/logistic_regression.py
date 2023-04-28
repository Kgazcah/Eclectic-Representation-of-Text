from sklearn.linear_model import LogisticRegression
from classifier_abstract import Classifier


class LR(Classifier):
	def __init__(self, penalty='l2', tol=0.0001, C=1.0, solver='lbfgs', max_iter=100):
		"""
		Parameters
		---------------
		penalty : l1, l2, elasticnet, None, default='l2'
			None: no penalty is added;
			'l2': add a L2 penalty term and it is the default choice
			'l1': add a L1 penalty term;
			'elasticnet': both L1 and L2 penalty terms are added.

		tol : float, default=1e-4
			Tolerance for stopping criteria.
		
		C : float, default=1.0
			Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

		solver : 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'; default='lbfgs'
			Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
			https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

		max_iter : int, default=100
			Maximum number of iterations taken for the solvers to converge.
		"""
		self.classifier=LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver, max_iter=max_iter)
	
	def fit(self, X_train, y_train):
		super().fit(X_train, y_train)
		
	def predict(self, X_test):
		y_pred = super().predict(X_test)
		return y_pred


	def metrics(self, y_test, y_pred):
		super().metrics(y_test, y_pred)

