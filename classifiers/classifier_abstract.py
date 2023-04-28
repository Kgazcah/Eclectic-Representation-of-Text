from abc import ABC, abstractmethod
from sklearn.metrics import (accuracy_score, confusion_matrix, 
							ConfusionMatrixDisplay, precision_score,
							recall_score, f1_score)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#Abstraction of classifiers
class Classifier(ABC):
	def fit(self, X_train, y_train):
		self.classifier = self.classifier.fit(X_train, y_train)


	def predict(self, X_test):
		y_pred = self.classifier.predict(X_test)
		return y_pred

	def metrics(self, y_test, y_pred):
		cm = confusion_matrix(y_test, y_pred)
		acc = accuracy_score(y_test, y_pred)
		# pre = precision_score(y_test, y_pred)
		re = recall_score(y_test, y_pred)
		# f1 = f1_score(y_test, y_pred)
		spe = cm[0,0]/(cm[0,1]+cm[0,0])

		print(f'The accuracy score of {self.classifier} is {round(acc,3)}, the recall score is {round(re,3)}, the speficity score is {round(spe,3)} and the confusion matrix is \n {cm}') # 
		# plt.figure(1)
		x = ["Acc","Re","Spe"]
		y = [acc,re,spe]
		plt.text(0, 0.1, f'{round(acc,3)}')
		plt.text(1, 0.1, f'{round(re,3)}')
		plt.text(2, 0.1, f'{round(spe,3)}')
		plt.bar(x,y, color = 'firebrick')

		# plt.figure(2)
		cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
		cm_display.plot()
		

		plt.show()


	# def save(self):
	# 	pass
