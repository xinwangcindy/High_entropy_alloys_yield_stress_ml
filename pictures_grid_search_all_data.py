from Tools import *
import scipy.misc
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ConstantKernel)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, explained_variance_score
import time
import matplotlib.pyplot as plt

subtract_ys_test = False
increased_ys = False
scale_data = False
k_fold = True
input_pca = True

if subtract_ys_test and increased_ys:
    raise Exception('If you want to subtract the YS, the increased_ys should be False')
elif subtract_ys_test and scale_data:
    raise Exception('If you want to subtract the YS, the scale_data should be False')

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)  # increased_ys=True

X, Y = data.get_pictures_data()
scaler = preprocessing.StandardScaler()
X_scale = scaler.fit_transform(copy.deepcopy(X))
#Number of inputs to train:
test_size = 0.
numberOfPCAComponents = 1
stateNumber = 22



if input_pca:
    X, eigenvalues, eigen_vectors,explained_variance = pca_input(numberOfPCAComponents, X_scale)  # PCA as input
# In[]: Save images generated by PCA
    #for num_sample, sample in enumerate(eigen_vectors):
        #sample = np.reshape(sample, (256, 256))
        #scipy.misc.imsave('pca_pictures/' + str(num_sample) + '.jpg', sample)
#Plotting yield stress with respect to components
plt.plot(X[:,0],Y,'r.')
plt.show()
'''

X, testX, Y, testY = split_train_test(X, Y, test_size, scale_data=True)

id_test = get_id_test(increased_ys, scale_data, input_pca=input_pca, comment='pictures')
plots = Plots(id_test, increased_ys=increased_ys)  # increased_ys=True

# Plot features with PCA(3 components)
# plots.pca_three_components(X, Y)

# Plot features with PCA(n components)
# plots.pca_ncomponents(X, Y, 2)

# Other techniques
# plots.unsupervised_models(X, Y)

all_results = []
all_predictions = []


def optimize_model(clf, param_grid, name):
    start_time = time.time()
    print('#####' + name + '#####')
    scoring = {'MAE':'neg_mean_absolute_error',
            'MSE':'neg_mean_squared_error',
            'MAPE': make_scorer(mean_absolute_percentage_error),
            'R2':'r2'}
    model = GridSearchCV(clf, cv=10, param_grid=param_grid,scoring=scoring,
            refit='MAE', return_train_score=True )
    model.fit(X,Y)
    results = model.cv_results_
    for scorer in scoring:
        sample_score_mean = results['mean_test_%s' % (scorer)]
        best_index = np.nonzero(results['rank_test_MAE'] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
        print(name + "_best_mean_test_%s" % scorer, best_score)
    time_computation = (time.time() - start_time)
    #    print('Time:',round(time_computation), 'seconds')
    #    print(predY[:3])
    print('      Training done in ', time_computation, 'sec')
    print('Best parameters: ' + str(model.best_params_))
    print()
    return model

# In[]: Gaussian Process Regressor
param_grid_GPR = {"kernel": [RationalQuadratic(l) 
                        for l in np.logspace(-2, 2, 10)],
                    'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}
clf = GaussianProcessRegressor(random_state = stateNumber)

GPRRQ = optimize_model(clf, param_grid_GPR, 'GaussianProcessRegressorRQ')


#GP using RBF
param_grid_GPR = {"kernel": [RBF(l) 
                        for l in np.logspace(-2, 2, 10)],
                    'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}

GPRRBG = optimize_model(GaussianProcessRegressor(), param_grid_GPR,
        'GaussianProcessRegressorRBF')
#GP Using Matern
param_grid_GPR = {"kernel": [Matern(l) 
                        for l in np.logspace(-2, 2, 10)],
                  'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}

GPRMatern = optimize_model(GaussianProcessRegressor(), param_grid_GPR,
        'GaussianProcessRegressorMatern')


# In[]: Gradient Boosting Regressor
param_grid_GBR = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'learning_rate': [10, 1, 0.1, 0.01],
                  'n_estimators': [80, 100, 120],
                  'max_depth': [3, 5, 7]}
clf = GradientBoostingRegressor(random_state=stateNumber)
GBR = optimize_model(GradientBoostingRegressor(), param_grid_GBR, 'GradientBoostingRegressor')


# In[]: Decision Tree Regressor
param_grid_DTR = {'criterion': ['mse', 'mae'],
                  'max_depth': [3, 5, 7]}
clf = DecisionTreeRegressor(random_state=stateNumber)
DTR = optimize_model(clf, param_grid_DTR, 'DecisionTreeRegressor')

# In[]: BayesianRidge
param_grid_BR = {'n_iter': [100, 200, 300, 400],
                 'tol': [1e-3, 1e-3, 1e-4],
                 'alpha_1': [1e-05, 1e-06, 1e-07],
                 'lambda_1': [1e-05, 1e-06, 1e-07],
                 'fit_intercept': [True, False],
                 'normalize': [True, False]}
clf = BayesianRidge()
BR = optimize_model(clf, param_grid_BR, 'BayesianRidge')

# In[]: KNeighbors Regressor

param_grid_KNR = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8],
                  'weights': ['uniform', 'distance']}

clf = KNeighborsRegressor()

KNR = optimize_model(clf, param_grid_KNR, 'KNeighborsRegressor')


# In[]: Kernel Ridge Regression
param_grid_KR = {"alpha": [0.01, 0.001, 0.0001, 0.00001],
                 'kernel': ['linear', 'rbf', 'poly'],
                 "gamma": [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                 'degree': [2, 3]}
clf = KernelRidge()
KR = optimize_model(clf, param_grid_KR, 'KernelRidgeRegression')
'''


