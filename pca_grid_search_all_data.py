from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ConstantKernel)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, explained_variance_score
import time
from Tools import *
from sklearn.preprocessing import scale

data = Dataset(increased_ys = False)  # increased_ys=True

X, Y = data.get_estimated_data()
#X, Y = data.get_no_estimated_data()
numberOfComponents = 3


#Number of inputs to train:
stateNumber = 22
CVCount = 10
all_results = []
all_predictions = []
#Scaling Data
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(copy.deepcopy(X))
#Run PCA on scaled data
X, eigenvalues, eigen_vectors, explained_variance = pca_input(numberOfComponents, X)
print('explained variance is: ', explained_variance)



def optimize_model(clf, param_grid, name):
    start_time = time.time()
    print('#####' + name + '#####')
    scoring = {'MAE':'neg_mean_absolute_error',
            'MSE':'neg_mean_squared_error',
            'MAPE': make_scorer(mean_absolute_percentage_error),
            'R2':'r2'}
    model = GridSearchCV(clf, cv=CVCount, param_grid=param_grid,scoring=scoring,
            refit='MAE', return_train_score=True )
    model.fit(X,Y)
    results = model.cv_results_
    for scorer in scoring:
        sample_score_mean = results['mean_test_%s' % (scorer)]
        best_index = np.nonzero(results['rank_test_MAE'] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
        print(name + "mean_test_%s" %scorer, best_score)
       # print(name + "_best_mean_test_%s" % scorer, sample_score_mean)
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


