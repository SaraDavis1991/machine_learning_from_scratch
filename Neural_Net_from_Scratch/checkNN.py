import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt 
from nn import build_model
from nn import plot_decision_boundary
from nn import predict
from nn import build_model_691
np.random.seed(0)
X, y = make_moons(200, noise = 0.20)
plt.scatter(X[:,0], X[:, 1], s = 40, c= y, cmap = plt.cm.Spectral) #shows original scatter
plt.figure(figsize = (16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
	plt.subplot(2, 2, i +1)
	plt.title ('HiddenLayerSize%d' % nn_hdim)
	model = build_model(X, y, nn_hdim)
	plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()

np.random.seed(0)
X, y = make_blobs(n_samples = 100, centers = 3, n_features = 2, random_state = 0)
plt.scatter(X[:, 0], X[:, 1], s = 40, c = y, cmap = plt.cm.Spectral)
plt.figure (figsize =( 16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
	plt.subplot(2, 2, i +1)
	plt.title ('HiddenLayerSize%d' % nn_hdim)
	model = build_model_691(X, y, nn_hdim)
	plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()


