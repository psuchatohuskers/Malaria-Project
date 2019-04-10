import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import get_data, sigmoid, sigmoid_cost, error_rate, display_img


class LogisticRegression(object):
	def __init__(self):
		pass

	def fit(self, X, Y, learning_rate=10e-7, l2 = 0, l1 = 0, epochs=100, show_fig=False):
		X, Y = shuffle(X,Y)
		N1 = X.shape[0]
		test_split = np.round(N1*0.8).astype(int)
		Xvalid, Yvalid = X[test_split:], Y[test_split:]
		X, Y = X[:test_split], Y[:test_split]


		N, D = X.shape
		self.W = np.random.rand(D) / np.sqrt(D)
		self.b = 0

		costs = []
		best_validation_error = 1
		for i in range(epochs):
			pY = self.forward(X)

			#gradient descent
			self.W -= learning_rate*(X.T.dot(pY-Y) + l2*self.W + l1*np.sign(self.W))
			self.b -= learning_rate*((pY-Y).sum() + l2*self.b + l1*np.sign(self.b))
			if i%20 == 0:
				pYvalid = self.forward(Xvalid)
				c = sigmoid_cost(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.round(pYvalid))
				print("i:",i,"cost:",c,"error:",e)
				if e < best_validation_error:
					best_validation_error = e
		print("best_validation_error:",best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self,X):
		return sigmoid(X.dot(self.W) + self.b)

	def predict(self, X):
		pY = self.forward(X)
		return np.round(pY)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)
def main():
	PATH_INFECT = "cell_images/Parasitized/"
	PATH_UNINFECT = "cell_images/Uninfected/"
	label = ["Uninfected","Infected"]
	data = get_data(PATH_INFECT,PATH_UNINFECT)
	data = shuffle(data)
	# for i in range(4):
	# 	display_img(data[i,1:],label[data[i,0].astype(int)])
	Y = data[:,0].astype(np.float128)
	X = data[:,1:].astype(np.float128)
	model = LogisticRegression()
	model.fit(X,Y, epochs = 1000,show_fig = True)


if __name__ == '__main__':
	main()




