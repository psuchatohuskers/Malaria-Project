import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import get_data, sigmoid, sigmoid_cost, error_rate, display_img

def forward(X,W,b):
	return sigmoid(X.dot(W) + b)

def predict(X,W,b):
	pY = forward(X,W,b)
	return np.round(pY)

PATH_INFECT = "cell_images/Parasitized/"
PATH_UNINFECT = "cell_images/Uninfected/"
label = ["Uninfected","Infected"]
data = get_data(PATH_INFECT,PATH_UNINFECT)
data = shuffle(data)
Y = data[:,0]
X = data[:,1:]
X, Y = shuffle(X,Y)
N1 = X.shape[0]
test_split = np.round(N1*0.8).astype(int)
Xvalid, Yvalid = X[test_split:], Y[test_split:]
plt.hist(Yvalid)
plt.show()
X, Y = X[:test_split], Y[:test_split]
learning_rate = 0.01
N, D = X.shape
W = np.random.rand(D) / np.sqrt(D)
b = 0
pY = forward(X,W,b)
print(pY)
print(pY.sum())
W -= learning_rate*(X.T.dot(pY-Y))
b -= learning_rate*((pY-Y).sum())
pYvalid = forward(Xvalid,W,b)
plt.hist(sigmoid(Xvalid.dot(W) + b))
plt.show()
print(pYvalid.sum())
# c = sigmoid_cost(Yvalid, pYvalid)
# e = error_rate(Yvalid, np.round(pYvalid))
# print(c)
# print(e)