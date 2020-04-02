from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ConstantKernel)
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, explained_variance_score
import time
import sys
from Tools import *

data = Dataset(increased_ys = False)  # increased_ys=True

#X, Y = data.get_estimated_data()
X, Y = data.get_no_estimated_data()

#Number of inputs to train:
test_size = 0.
stateNumber = 22
CVCount = 10
all_results = []
all_predictions = []
#Fitting data
SFE_std = [39.0,12.0]
numberOfPoints = 100
region_size_input = np.linspace(0.1,12,numberOfPoints)
region_size_input = region_size_input.reshape(-1,1)
for j in range(0,len(SFE_std)):
    if j == 0:
        SFE_mean = [127.1, 84.7, 72.]
    else:
        SFE_mean = [35.]
    for i in range(0,len(SFE_mean)):
        sfe_energy = np.ones((numberOfPoints,1)) * SFE_mean[i]
        std_array = np.ones((numberOfPoints,1)) * SFE_std[j]
        tempX = np.concatenate((sfe_energy, std_array), axis = 1)
        tempX = np.concatenate((tempX, region_size_input), axis = 1)
        if i == 0 and j == 0:
            testX = tempX
        else:   
            testX = np.concatenate((testX, tempX), axis = 0)
np.savetxt('results_prediction/testInputData.dat', testX)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)
X, tempX, Y, tempY = split_train_test(X, Y, test_size, scale_data=False)


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
    time_computation = (time.time() - start_time)
    #    print('Time:',round(time_computation), 'seconds')
    print('      Training done in ', time_computation, 'sec')
    print('Best parameters: ' + str(model.best_params_))
    print()
    return model
model = joblib.load('prescribed_GPRRBF.pkl')
predY = model.predict(testX)
np.savetxt('results_prediction/testOutputData.dat', predY)
'''
# In[]: Gradient Boosting Regressor
param_grid_GBR = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'learning_rate': [10, 1, 0.1, 0.01],
                  'n_estimators': [80, 100, 120],
                  'max_depth': [3, 5, 7]}
clf = GradientBoostingRegressor(random_state=stateNumber)

model = optimize_model(GradientBoostingRegressor(), param_grid_GBR, 'GradientBoostingRegressor')

# In[]: Gaussian Process Regressor
param_grid_GPR = {"kernel": [RationalQuadratic(l) 
                        for l in np.logspace(-2, 2, 10)],
                    'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}
clf = GaussianProcessRegressor(random_state = stateNumber)

model = optimize_model(clf, param_grid_GPR, 'GaussianProcessRegressorRQ')
joblib.dump(model.best_estimator_, 'prescribed_GPRRQ.pkl', compress = 1)


#GP using RBF
param_grid_GPR = {"kernel": [RBF(l) 
                        for l in np.logspace(-2, 2, 10)],
                    'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}

model = optimize_model(GaussianProcessRegressor(), param_grid_GPR,
        'GaussianProcessRegressorRBF')
joblib.dump(model.best_estimator_, 'prescribed_GPRRBF.pkl', compress = 1)
#GP Using Matern
param_grid_GPR = {"kernel": [Matern(l) 
                        for l in np.logspace(-2, 2, 10)],
                  'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001 ]}

model = optimize_model(GaussianProcessRegressor(), param_grid_GPR,
        'GaussianProcessRegressorMatern')
joblib.dump(model.best_estimator_, 'prescribed_GPRMatern.pkl', compress = 1)

# In[]: Decision Tree Regressor
param_grid_DTR = {'criterion': ['mse', 'mae'],
                  'max_depth': [3, 5, 7]}
clf = DecisionTreeRegressor(random_state=stateNumber)
model = optimize_model(clf, param_grid_DTR, 'DecisionTreeRegressor')


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

model = optimize_model(clf, param_grid_KNR, 'KNeighborsRegressor')
joblib.dump(model.best_estimator_, 'prescribed_KNR.pkl', compress = 1)

# In[]: Kernel Ridge Regression
param_grid_KR = {"alpha": [0.01, 0.001, 0.0001, 0.00001],
                 'kernel': ['linear', 'rbf', 'poly'],
                 "gamma": [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                 'degree': [2, 3]}
clf = KernelRidge()
model = optimize_model(clf, param_grid_KR, 'KernelRidgeRegression')
joblib.dump(model.best_estimator_, 'prescribed_KRR.pkl', compress = 1)
'''





