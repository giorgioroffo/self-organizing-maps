# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom

from pylab import bone, pcolor, colorbar, plot, show

# Importing the dataset
dataset = pd.read_csv('./dataset/dataset_bank_credit_card_applications.csv')

# SOM Size
n_rows = 15
n_cols = 10


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x = n_cols, y = n_rows, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', '+']
colors = ['r', 'b']

# For each observation take its BMU and visualize it
for i, x in enumerate(X):
    BMU = som.winner(x) # winning node is the Best Matching Unit BMU
    plot(BMU[0] + 0.5,
         BMU[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
# BMU(0,0) is the first node, the list of all the observations associated to it it's in the list
mappings = som.win_map(X) # In each winning node, we have the list of all observations associated with the BMU
id_x = np.argmax(np.max(som.distance_map().T, axis=0))
id_y = np.argmax(som.distance_map().T[:,id_x])
frauds = mappings[(id_x,id_y)]
frauds = sc.inverse_transform(frauds)

