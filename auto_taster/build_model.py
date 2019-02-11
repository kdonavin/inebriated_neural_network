import ipdb
import math
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sys
import preprocessor as pre
# import spacy
# from spacy.tokenizer import Tokenizer

#Statics

def fit_transform_comments(X_train, X_test):
	'''
	Vectorize a training and test set

	INPUT
	-----
	X_train: np.array on which to train the TFIDF Vectorizer
	X_test: np.array to vectorize based on TFIDF from X_train

	OUTPUT
	------
	lemmatized, tokenized document
	'''	
	assert 'comments' in X_train.columns, \
		"X_train must have a column named comments"
	assert 'comments' in X_test.columns, \
		"X_test must have a column named comments"

	# nlp = spacy.load('en')	
	tfidf = TfidfVectorizer()
	vectors_train =  tfidf.fit_transform(X_train.comments)
	vectors_test = tfidf.transform(X_test.comments)
	vectors_train = pd.DataFrame(vectors_train.toarray(), \
		index = X_train.index, columns=tfidf.get_feature_names())
	vectors_test = pd.DataFrame(vectors_test.toarray(), \
		index = X_test.index, columns=tfidf.get_feature_names())
	return vectors_train, vectors_test, tfidf

def balance_not_hot(X, y, minority_class=1):
	'''
	A method to balance minority 'Not-Hot' class.

	INPUT
		X - iterable 2D array-like object
		y - iterable 1D NumPy array
		minority_class - value that represents the minority class

	OUTPUT - minority-class balanced DataFrame object
	'''

	not_hot_condition = (y == minority_class)
	not_hot_count = not_hot_condition.sum()
	ttb_count = (not_hot_condition == False).sum()
	assert ttb_count > not_hot_count, "Error: minority class must be fewer than other class"
	X_new, y_new = resample(X[not_hot_condition], y[not_hot_condition], \
		n_samples=(ttb_count - not_hot_count))
	return pd.concat((X, X_new)), np.concatenate((y, y_new))

def print_test_report(estimator, X_test, y_test):
	'''
	Method to print sklearn's metric.classification report from an estimator
	and test data
	'''

	y_hat = estimator.predict(X_test)
	
	print('accuracy score : {}'.format(estimator.score(X_test, y_test)))
	
	report = metrics.classification_report(y_true=y_test, y_pred=y_hat)
	print(report)

###############################################################################

class FillNulls(BaseEstimator, TransformerMixin):
	'''
	Fill Nulls with -1 after scaling in order to avoid influencing scale
	'''
	def fit(self, X, y):
		return self

	def transform(self, X):
		X.replace(np.nan, stats.norm().ppf(0.001), inplace=True) #very unlikely value in standard normal
		return X

class ScaleNaNs(BaseEstimator, TransformerMixin):
	'''
	Class to standardize features while ignoring NaNs
	'''
	def fit(self, X, y):
		self.means = X.mean(axis=0)
		self.stds = X.std(axis=0)
		return self

	def transform(self, X):
		X = X.subtract(self.means)/self.stds
		return X

def scale_remove_nulls(X):
	##Scale & Fill Nulls
	scale = ScaleNaNs()
	scale.fit(X, y=None)
	X = scale.transform(X)
	fill = FillNulls()
	X = fill.transform(X)
	return X

if __name__ == '__main__':

	#Allow specification of filename to save grid searched model to.
	if len(sys.argv) > 1:
		 model_prefix = sys.argv[1]
	else:
		model_prefix = 'best'

	#Datasets
	X, y = pre.load_data_batch_level()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, \
		random_state=1234) 

	##Scale & Fill Test Nulls - SHOULDN'T I USE training standards
	X_test = scale_remove_nulls(X_test)

	#Pipeline
	tree_pipe = Pipeline([
		('fillnulls', FillNulls()),
		('trees', RandomForestClassifier())
	])
	mlp_pipe = Pipeline([
		# ('scale', ScaleNaNs()),
		# ('fillnulls', FillNulls()),
		('mlp', MLPClassifier())
	])

	mlp_grid_params = [
		{
			'mlp__alpha': [0.001],
			'mlp__early_stopping': [True],
			'mlp__hidden_layer_sizes': [\
				(128, 128, 128, 128, 128, 128, 128), \
				(128, 128, 128, 128, 64, 32, 16), \
			],
			'mlp__learning_rate_init': [0.01, 0.001],
			'mlp__max_iter': [1500, 1000],
			'mlp__solver': ['adam'] 
		}
	]

	tree_grid_params = [
		{
			'trees__n_estimators': np.linspace(50, 700, 14).astype('int').tolist()
		}
	]

	#Rebalance 
	X_balanced, y_balanced = balance_not_hot(X_train, y_train)

	#Tree Classifier
	# tree_grid = GridSearchCV(tree_pipe, tree_grid_params, scoring='recall', verbose=3)
	# tree_grid.fit(X_balanced, y_balanced)
	# print_test_report(tree_grid.best_estimator_, X_test, y_test)	
	# with open('src/best_tree.pkl', 'wb') as p:
		# pickle.dump(tree_grid.best_estimator_.steps[-1][-1], p)

	##MLP Classifier
	X_train = scale_remove_nulls(X_train) #REMOVE, leakage
	X_balanced, y_balanced = balance_not_hot(X_train, y_train)
	mlp_grid = GridSearchCV(mlp_pipe, mlp_grid_params, scoring='f1', verbose=3)
	mlp_grid.fit(X_balanced, y_balanced)
	print_test_report(mlp_grid.best_estimator_, X_test, y_test)
	best_pipe = mlp_grid.best_estimator_
	with open(model_prefix + '_mlp.pkl', 'wb') as p:
		pickle.dump(best_pipe, p)
