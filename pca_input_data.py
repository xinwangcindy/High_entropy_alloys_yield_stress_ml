from Tools import *
from pictures import TrainModels

subtract_ys_test = False
increased_ys = False
scale_data = True
# Prescribed Data
X, Y = data.get_no_estimated_data()
X_new, eigenvalues, eigen_vectors, explained_variance = pca_input(num_features, copy.deepcopy(X))
#Estimated Data
X, Y = data.get_estimated_data()
X_new, eigenvalues, eigen_vectors, explained_variance = pca_input(num_features, copy.deepcopy(X))
# Picture Data
data = Dataset(increased_ys=increased_ys, scale_data=scale_data)
X, Y = data.get_pictures_data()
X_new, eigenvalues, eigen_vectors, explained_variance = pca_input(num_features, copy.deepcopy(X))

