import numpy as np


def linear_svm(W, b, x, y):
	
	linear_loss = 0.0
	final_scores = x.dot(W)
	y_scores = final_scores[np.arange(final_scores.shape[0]),y]
	margins = np.maximum(0,final_scores - np.matrix(y_scores).T + 1)
	margins[np.arange(x.shape[0]),y] = 0
	linear_loss = np.mean(np.sum(margins,axis = 1))
	return linear_loss

def logistic_regression(W, b, x, y):
	logistic_loss = 0.0
	final_scores = x.dot(W)
	final_scores -= np.max(final_scores)
	final_scores_expon = np.exp(final_scores)
	final_scores_expon_corr = final_scores_expon[range(x.shape[0]),y]
	final_scores_sum = np.sum(final_scores_expon,axis=1)
	#import IPython
	#IPython.embed()
	logistic_loss = -np.sum(np.log(final_scores_expon_corr/final_scores_sum))
	logistic_loss /= x.shape[0]
	return logistic_loss