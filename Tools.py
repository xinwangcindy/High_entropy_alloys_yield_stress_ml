import copy
import numpy as np
import pandas as pd
import pickle
import random
import re
import time
from random import shuffle
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from IPython.display import display
from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    median_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from constants import *
from scipy.spatial import distance
from decorators import *
#from algorithms.constants import *
#from algorithms.decorators import *

constant = Constants()


def open_variable(directory):
    """
    Using the filename import a variable.
    Args:
        directory: string with the filename
    Return:
         variable
    Example:
        open_variable('all_pictures')
    """
    f = open(directory + '.pckl', 'rb')
    variable = pickle.load(f)
    f.close()
    return variable


def save_variable(variable, name):
    """
    Save a variable into pckl format.
    Args:
        variable: file to save
        name: string with the filename
    Example:
        save_variable(pictures, 'all_pictures')
    """
    f = open(name + '.pckl', 'wb')
    pickle.dump(variable, f)
    f.close()


def convert_to_increased_YS(X, Y):
    """
    Subtract a part of the yield stress depending of the gamma.
    Args:
        X: Array with gamma, std(gamma) and the grain size.
        Y: Original yield stress array.
    Return:
        Y_inc: new array with the increased yield stress.
    """
    Y_inc = copy.deepcopy(Y)
    for num_sample, sample in enumerate(X):
        if str(sample[0])[0] == '1':
            Y_inc[num_sample] = Y_inc[num_sample] - 145
        elif str(sample[0])[0] == '3':
            Y_inc[num_sample] = Y_inc[num_sample] - 400
        elif str(sample[0])[0] == '7' or str(sample[0])[0] == '6':
            Y_inc[num_sample] = Y_inc[num_sample] - 235
        elif str(sample[0])[0] == '8':
            Y_inc[num_sample] = Y_inc[num_sample] - 220
        else:
            Exception('The inputs can not be recognized as one of the gammas. They can not be scaled')
    return Y_inc


class Dataset:
    def __init__(self, increased_ys=False, scale_data=False):
        self.increased_ys = increased_ys
        self.scale_data = scale_data
        self.mean_d = None
        self.std_isf = None
        self.all_pictures = None
        self.all_id_pictures = None
        self.mean_isf = None
        self.number_nonRepeatedValues = None
        self.Y_estimated = None
        self.X_pictures = None
        self.Y_estimated_inc = None
        self.x_non_est = None
        self.init_import_data()

    def get_variables(self):
        """
        Import saved variables.
        Return:
            all_pictures: variable with all the pictures data in a matrix.
            all_id_pictures: variable with the name of each file.
        """
        self.all_pictures = open_variable('variables/allPictures_Clean')
        self.all_id_pictures = open_variable('variables/all_idPictures_Clean')
        self.all_pictures = np.asarray(self.all_pictures)
        self.all_id_pictures = np.asarray(self.all_id_pictures)

    def info_from_filename(self):
        """
        Obtain information of each picture depending of the filename.
        Args:
            all_id_pictures: matrix with the name of each sample.
        Return:
            mean_d: array with the grain size of each sample.
        """
        self.mean_d = []
        for id_picture in self.all_id_pictures:
            obtain_id = id_picture[0].split('_')
            # gamma_value=int(re.search(r'\d+', obtain_id[0]).group())
            diameter_value = int(re.search(r'\d+', obtain_id[1]).group())
            # simulation_value = int(re.search(r'\d+', obtain_id[2]).group())
            if diameter_value == 5:
                diameter_value = 0.5
            elif diameter_value == 25:
                diameter_value = 0.25
            elif diameter_value == 4:
                diameter_value = 4.5
            self.mean_d.append(diameter_value)
        self.mean_d = np.asarray(self.mean_d)

    def init_import_data(self):
        """"
        Common process to initialze the data.
        Args:
            all_pictures: matrix with all the pictures as vectors.
            all_id_pictures: the second column has the yield stress of each sample.
        Returns:
            X_pictures: matrix mxn where m is the number of samples and n the total of pixels (256x256).
            Y_estimated: m vector with the yield stress of each sample.
        """
        self.get_variables()
        self.info_from_filename()
        self.X_pictures = self.all_pictures.astype(np.float)
        self.Y_estimated = self.all_id_pictures[:, 1]
        self.Y_estimated = self.Y_estimated.astype(np.float)

    @timer
    def get_estimated_data(self):
        """"
        Method that returns the estimated inputs and outputs of the machine learning system.
        The data is obtained through the filename of the pictures.
        Args:
            X_pictures: matrix mxn where m is the number of samples and n the total of pixels (256x256).
            Y_estimated: m vector with the yield stress of each sample.
        Returns:
            x_estimated: mx3 with gamma, std(gamma) and the number of repeated pixels.
            Y_estimated: m vector with the yield stress.
            Y_estimated_inc: m vector with the increased yield stress.
        """
        self.number_nonRepeatedValues = []
        min_max_scaler = preprocessing.MinMaxScaler()
        for row in min_max_scaler.fit_transform(copy.deepcopy(self.X_pictures)):
            unique_value = len(np.unique(row))
            self.number_nonRepeatedValues = np.append(self.number_nonRepeatedValues, unique_value)
        pictures_matrix = self.all_pictures.astype(np.float)
        self.std_isf = pictures_matrix.std(1)
        self.mean_isf = pictures_matrix.mean(1)
        x_estimated = np.asarray([self.mean_isf, self.std_isf, self.number_nonRepeatedValues])
        x_estimated = x_estimated.transpose((1, 0))
        if self.increased_ys:
            self.Y_estimated_inc = convert_to_increased_YS(x_estimated,
                                                           copy.deepcopy(self.Y_estimated))  # Increased Yield Stress
            if self.scale_data:
                x_estimated = min_max_scaler.fit_transform(x_estimated)
            return x_estimated, self.Y_estimated_inc
        else:
            if self.scale_data:
                x_estimated = min_max_scaler.fit_transform(x_estimated)
            return x_estimated, self.Y_estimated

    @timer
    def get_no_estimated_data(self):
        """"
        Method that returns the calculated inputs and outputs of the machine learning system.
        The data is obtained through the dataset csv.
        Returns:
            x_non_est: mx3 with gamma, std(gamma) and the grain size.
            y_non_est: m vector with the yield stress.
            y_non_est_inc: m vector with the increased yield stress.
        """
        df = pd.read_csv("data/dataset.csv")  # We do not have the isf 35 information
        self.x_non_est = df.drop(["YS"], axis=1)
        y_non_est = df["YS"]
        self.x_non_est = self.x_non_est .values
        y_non_est = y_non_est.values
        if self.increased_ys:
            y_non_est_inc = convert_to_increased_YS(self.x_non_est , y_non_est)  # Increased Yield Stress
            if self.scale_data:
                min_max_scaler = preprocessing.MinMaxScaler()
                x_non_est = min_max_scaler.fit_transform(copy.deepcopy(self.x_non_est))
                return x_non_est, y_non_est_inc
            else:
                return self.x_non_est, y_non_est_inc
        else:
            if self.scale_data:
                min_max_scaler = preprocessing.MinMaxScaler()
                x_non_est = min_max_scaler.fit_transform(copy.deepcopy(self.x_non_est))
                return x_non_est, y_non_est
            else:
                return self.x_non_est, y_non_est

    @timer
    def get_pictures_data(self):
        """"
        Method that returns the inputs and outputs of the machine learning system.
        Returns:
            X_pictures: matrix mx(256x256) with the pictures as vector.
            Y_estimated: m vector with the yield stress.
            y_estimated_inc: m vector with the increased yield stress.
        """
        if self.increased_ys:
            mean_gamma_row = []
            for row in self.X_pictures:
                mean_gamma_row.append([np.mean(row)])
            y_estimated_inc = convert_to_increased_YS(mean_gamma_row, self.Y_estimated)  # Increased Yield Stress
            if self.scale_data:
                min_max_scaler = preprocessing.MinMaxScaler()
                x_pic_scale = min_max_scaler.fit_transform(copy.deepcopy(self.X_pictures))
                return x_pic_scale, y_estimated_inc
            else:
                return self.X_pictures, y_estimated_inc
        else:
            if self.scale_data:
                min_max_scaler = preprocessing.MinMaxScaler()
                x_pic_scale = min_max_scaler.fit_transform(copy.deepcopy(self.X_pictures))
                return x_pic_scale, self.Y_estimated
            else:
                return self.X_pictures, self.Y_estimated

    def add_column_int_gamma(self, input):
        """"
        Method that appends one column with the gammas as int to the input array.
        """
        energies_clean = []
        for num_energy, energy in enumerate(input[:, 0]):
            if int(str(energy)[:1]) == 1:
                energies_clean = np.append(energies_clean, 127)
            elif int(str(energy)[:1]) == 3:
                energies_clean = np.append(energies_clean, 35)
            elif int(str(energy)[:1]) == 7 or int(str(energy)[:1]) == 6:
                energies_clean = np.append(energies_clean, 72)
            elif int(str(energy)[:1]) == 8:
                energies_clean = np.append(energies_clean, 84)
        energies_clean = np.reshape(energies_clean, (len(energies_clean), 1))
        output = np.column_stack((input, energies_clean))
        return output


def reduce_pixels_random(x_scale, num_pixels):
    """"
    Method that returns a new matrix reduced.
    Args:
        x_scale: input matrix scaled.
        num_pixels: number of features (columns) desired.
    Returns:
        X_pictures: matrix mx(256x256) with the pictures as vector.
        Y_estimated: m vector with the yield stress.
        y_estimated_inc: m vector with the increased yield stress.
    """
    random.seed(0)
    new_x = []
    for row in x_scale:
        new_values = random.sample(set(row), num_pixels)
        new_x = np.append(new_x, new_values)
    new_x = np.reshape(new_x, (int(len(new_x) / num_pixels), num_pixels))
    # X_scale=np.delete(X_scale,np.s_[100:], axis=1)
    return new_x


'''
def getCovarianceMatrix(X_scale):
    mean_vec = np.mean(X_scale, axis=0)
    cov_mat = (X_scale - mean_vec).T.dot((X_scale - mean_vec)) / (X_scale.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)
    return cov_mat

def getNumpyCovarianceMatrix(X_scale):
    cov_mat=np.cov(X_scale.T)
    print('NumPy covariance matrix:',cov_mat)
    return cov_mat

def getEigVectVal(X_scale):
    cov_mat=np.cov(X_scale.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    return cov_mat,eig_vals, eig_vecs

def getCorrelMatrix(X_scale):
    cor_mat1 = np.corrcoef(X_scale.T)
    return cor_mat1

def sortInfluenceEigenvectors(eig_vals,eig_vecs):
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
    
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
'''


def split_energies(X, Y, gamma=None):
    """"
    Method that returns lists of single gammas.
    Args:
        X: all the input data.
        Y: vector with the yield stress.
        gamma: desired gamma to split.
    Returns:
        X_list_: list with the inputs of the desired energy.
        Y_list_: list with the yield stress of the desired energy.
    """
    X_list_127 = []
    X_list_84 = []
    X_list_72 = []
    X_list_35 = []
    Y_list_127 = []
    Y_list_84 = []
    Y_list_72 = []
    Y_list_35 = []
    for num_sample, sample in enumerate(X):
        if str(sample[0])[0] == '1':
            X_list_127 = np.append(X_list_127, sample)
            Y_list_127 = np.append(Y_list_127, Y[num_sample])
        elif str(sample[0])[0] == '8':
            X_list_84 = np.append(X_list_84, sample)
            Y_list_84 = np.append(Y_list_84, Y[num_sample])
        elif str(sample[0])[0] == '7':
            X_list_72 = np.append(X_list_72, sample)
            Y_list_72 = np.append(Y_list_72, Y[num_sample])
        elif str(sample[0])[0] == '3':
            X_list_35 = np.append(X_list_35, sample)
            Y_list_35 = np.append(Y_list_35, Y[num_sample])

    X_list_127 = np.reshape(X_list_127, (int(len(X_list_127) / 3), 3))
    X_list_84 = np.reshape(X_list_84, (int(len(X_list_84) / 3), 3))
    X_list_72 = np.reshape(X_list_72, (int(len(X_list_72) / 3), 3))
    X_list_35 = np.reshape(X_list_35, (int(len(X_list_35) / 3), 3))
    if gamma == 127:
        return X_list_127, Y_list_127
    elif gamma == 84:
        return X_list_84, Y_list_84
    elif gamma == 72:
        return X_list_72, Y_list_72
    elif gamma == 35:
        return X_list_35, Y_list_35
    elif gamma is None:
        return X_list_127, Y_list_127, X_list_84, Y_list_84, X_list_72, Y_list_72, X_list_35, Y_list_35
    else:
        print('Any gamma has been specified--> returning original X,Y')
        return X, Y


def split_train_test(X, Y, test_size, scale_data=False):
    """"
    Method that split the original data into train and test.
    Args:
        X: all the input data.
        Y: vector with the yield stress.
        test_size: percentage of test --> from 0 to 1.
        scale_data: boolean to point out if a scaled is needed.
    Returns:
        trainX: input training values.
        trainY: output training values.
        testX: input testing values.
        testY: output testing values.
    """
    if scale_data:
        X = preprocessing.scale(copy.deepcopy(X))

    trainX, testX, trainY, testY = train_test_split(X, Y,
                                                    test_size=test_size,
                                                    random_state=None)
    return trainX, testX, trainY, testY


def pca_input(num_features, matrix):
    """"
    Method that transform the data to a projected matrix to change the input of the model.
    Args:
        num_features: num of components of the pca.
        matrix: original input data.
    Returns:
        matrix_projected: matrix with new axis.
        eigenvalues: eigenvalues.
        eigenvectors: eigenvectors.
    """
    pca = PCA(num_features)
    matrix_projected = pca.fit_transform(matrix)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    return matrix_projected, eigenvalues, eigenvectors, explained_variance_ratio


def listdic_to_array(list_dic):
    array = []
    for dicionary in list_dic:
        array.append(list(dicionary.values()))
    return array


def listStd(test, prediction):
    std_vector = []
    for num_sample, sample in enumerate(test):
        c = np.array([test[num_sample], prediction[num_sample]])
        std_vector = np.append(std_vector, np.std(c))
    return std_vector


def shuffle_list(train_x, train_y):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(train_x)))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(train_x[i])
        list2_shuf.append(train_y[i])

    list1_shuf = np.asarray(list1_shuf)
    list2_shuf = np.asarray(list2_shuf)
    return list1_shuf, list2_shuf


def list_std_percent(test, prediction):
    std_vector = []
    for num_sample, data_value in enumerate(prediction):
        std_row = (abs(data_value - test[num_sample])) / (np.sqrt(2) * data_value)
        std_vector = np.append(std_vector, std_row)
    return std_vector


# Normalization of the evaluation techniques
def root_mean_squared_error_norm(test, prediction):
    first_factor = []
    for num_sample, test_value in enumerate(test):
        c = (prediction[num_sample] - test_value) ** 2 / (prediction[num_sample]) ** 2
        first_factor = np.append(first_factor, c)
    RMSE_norm = 1 - np.sqrt(np.mean(first_factor))
    return RMSE_norm


def median_absolute_error_norm(test, prediction):
    first_factor = []
    for num_sample, test_value in enumerate(test):
        c = abs(prediction[num_sample] - test_value) / prediction[num_sample]
        first_factor = np.append(first_factor, c)
    MEDAE_norm = 1 - np.mean(first_factor)
    return MEDAE_norm



def mean_squared_log_error_norm(test, prediction):
    first_factor = []
    for num_sample, test_value in enumerate(test):
        c = (np.log(1 + prediction[num_sample]) - np.log(1 + test_value)) ** 2 / (
            np.log(1 + prediction[num_sample])) ** 2
        first_factor = np.append(first_factor, c)
    MSLE_norm = 1 - np.mean(first_factor)
    return MSLE_norm

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


class StandardModels:
    """
    Class with all the models usually used and already optimized.
    """

    def __init__(self):
        self.KNR = KNeighborsRegressor(n_neighbors=2, weights='distance')
        self.BRR = BayesianRidge(alpha_1=1e-07, lambda_1=1e-05, n_iter=100, tol=0.001)
        self.DTR = DecisionTreeRegressor(criterion='mse', max_depth=7)
        self.GBR = GradientBoostingRegressor(learning_rate=0.1, loss='ls', max_depth=3, n_estimators=80)
        self.KRR = KernelRidge(alpha=0.001, degree=2, gamma=1, kernel='rbf')
        self.GPR = GaussianProcessRegressor(alpha=0.01, kernel=Matern(length_scale=1, nu=1.5))
        self.SVRe = SVR(C=1000, degree=3, kernel='rbf', gamma=1)
        self.MLPR = MLPRegressor(activation='relu', hidden_layer_sizes=(100,), learning_rate='constant', max_iter=10000,
                                 solver='lbfgs')
        self.ABR = AdaBoostRegressor(learning_rate=1, loss='square', n_estimators=40)


class Models:
    """
    Class with all the necessary functions to train the model and evaluate the results.
    """

    def __init__(self, X, Y, models, subtract_ys=False, k_fold=False, test_size=None, insert_train_test=None):
        self.X = X
        self.Y = Y
        self.KNR = models.KNR
        self.BRR = models.BRR
        self.DTR = models.DTR
        self.GBR = models.GBR
        self.KRR = models.KRR
        self.GPR = models.GPR
        self.SVRe = models.SVRe
        self.MLPR = models.MLPR
        self.ABR = models.ABR
        self.models_used = None
        self.all_results = []
        self.all_predictions = []
        self.clf = None
        self.start_time = time.time()
        self.subtract_ys = subtract_ys
        self.k_fold = k_fold

        if test_size is None:
            self.test_size = constant.test_size
        else:
            self.test_size = test_size

        if insert_train_test is None:
            self.trainX, self.testX, self.trainY, self.testY = split_train_test(X, Y, self.test_size)
        else:
            self.trainX = insert_train_test[0]
            self.testX = insert_train_test[1]
            self.trainY = insert_train_test[2]
            self.testY = insert_train_test[3]
    @timer
    def train_model(self, clf):
        """"
        Method that train the model.
        Args:
            clf: model to train. Ex: GaussianProcessRegressor()
        Returns:
            clf: matrix with new axis.
            predY: eigenvalues.
        """
        self.clf = clf
        print('Training...')
        if self.k_fold is False:
            self.clf.fit(self.trainX, self.trainY)
            predY = self.clf.predict(self.testX).tolist()
            predY = np.resize(predY, (len(self.testY), 1))
            if self.subtract_ys is True:
                new_predY = convert_to_increased_YS(copy.copy(self.testX), copy.copy(predY))
                new_testY = convert_to_increased_YS(copy.copy(self.testX), copy.copy(self.testY))
                self.evaluate_model(testY=new_testY, predY=new_predY)
            else:
                self.evaluate_model(testY=self.testY, predY=predY)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=constant.num_splits_Kfold, random_state=None, shuffle=True)
            predictions = []
            tests = []
            for train_index, test_index in kf.split(self.X):
                trainX, testX = self.X[train_index], self.X[test_index]
                trainY, testY = self.Y[train_index], self.Y[test_index]
                self.clf.fit(trainX, trainY)
                predY = self.clf.predict(testX).tolist()
                predY = np.resize(predY, (len(testY), 1))
                if self.subtract_ys is True:
                    new_predY = convert_to_increased_YS(copy.copy(testX), copy.copy(predY))
                    new_testY = convert_to_increased_YS(copy.copy(testX), copy.copy(testY))
                    predictions.append(new_predY)
                    tests.append(new_testY)
                else:
                    predictions.append(predY)
                    tests.append(testY)
            self.evaluate_model(testY=tests, predY=predictions)

    def obtain_mean(self, process, testY, predY):
        results_eval_technique = []
        for num_test in range(len(testY)):
            results_eval_technique.append(process(testY[num_test], predY[num_test]))
        return np.mean(results_eval_technique)

    def evaluate_model(self, testY, predY, train_model=True):
        if self.k_fold:
            r2 = self.obtain_mean(r2_score, testY, predY)
            accuracy = 0
            MSE = self.obtain_mean(mean_squared_error, testY, predY)
            RMSE = np.sqrt(MSE)
            MAE = self.obtain_mean(mean_absolute_error, testY, predY)
            MSLE = self.obtain_mean(mean_squared_log_error, np.absolute(testY), np.absolute(predY))
            MEDAE = self.obtain_mean(median_absolute_error, testY, predY)
            MAPE = self.obtain_mean(mean_absolute_percentage_error, testY, predY)
            EVS = self.obtain_mean(explained_variance_score, testY, predY) * 100
            RMSE_norm = self.obtain_mean(root_mean_squared_error_norm, testY, predY) * 100
            MEDAE_norm = self.obtain_mean(median_absolute_error_norm, testY, predY) * 100
            MSLE_norm = self.obtain_mean(mean_squared_log_error_norm, testY, predY) * 100
            Mean_pred = 0
            CV = [0]
        else:
            r2 = r2_score(testY, predY)
            accuracy = self.clf.score(self.testX, testY) * 100
            MSE = mean_squared_error(testY, predY)
            RMSE = np.sqrt(mean_squared_error(testY, predY))
            MAE = mean_absolute_error(testY, predY)
            MSLE = mean_squared_log_error(abs(testY), abs(predY))
            MEDAE = median_absolute_error(testY, predY)
            MAPE = mean_absolute_percentage_error(testY, predY)
            EVS = explained_variance_score(testY, predY) * 100
            RMSE_norm = root_mean_squared_error_norm(testY, predY) * 100
            MEDAE_norm = median_absolute_error_norm(testY, predY) * 100
            MSLE_norm = mean_squared_log_error_norm(testY, predY) * 100
            Mean_pred = np.mean(predY)
            CV = cross_val_score(self.clf, self.trainX, self.trainY, cv=5)
        time_computation = (time.time() - self.start_time)
        results = {'r2[%]': r2, 'Accuracy[%]': accuracy, 'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE,
                   'MSLE': MSLE, 'MEDAE': MEDAE, 'MAPE[%]': MAPE, 'EVS[%]': EVS,
                   'RMSE_norm': RMSE_norm, 'MEDAE_norm': MEDAE_norm, 'MSLE_norm': MSLE_norm,
                   'CVmean': np.mean(CV) * 100, 'CVmax': max(CV) * 100,
                   'CVmin': min(CV) * 100,
                   'Time[sec]': time_computation, 'MeanPred': Mean_pred}
        self.all_results.append(results)
        self.all_predictions.append(predY)
        if train_model == False:
            return self.all_results

    def displayResults(self, matrix_all_results):
        values = {'r2[%]': matrix_all_results[:, 0],
                  'Accuracy[%]': matrix_all_results[:, 1],
                  'MSE': matrix_all_results[:, 2],
                  'RMSE': matrix_all_results[:, 3],
                  'MAE': matrix_all_results[:, 4],
                  'MSLE': matrix_all_results[:, 5],
                  'MEDAE': matrix_all_results[:, 6],
                  'MAPE[%]': matrix_all_results[:, 7],
                  'EVS[%]': matrix_all_results[:, 8],
                  'RMSE[%]': matrix_all_results[:, 9],
                  'MEDAE[%]': matrix_all_results[:, 10],
                  'MSLE[%]': matrix_all_results[:, 11],
                  'CVmean': matrix_all_results[:, 12],
                  'CVmax': matrix_all_results[:, 13],
                  'CVmin': matrix_all_results[:, 14],
                  'Time[sec]': matrix_all_results[:, 15],
                  'Mpred': matrix_all_results[:, 16]}

        df_all_results = pd.DataFrame(values, index=self.models_used)
        display(df_all_results)
        return df_all_results

    def execute_model(self):
        models = [self.KNR, self.BRR, self.DTR, self.GBR, self.KRR,
                  self.GPR, self.SVRe, self.MLPR, self.ABR]
        for clf in models:
            if clf is not None:
                self.train_model(clf)

    def get_results(self, train_model=True):
        if train_model:
            self.execute_model()
        matrix_all_results = listdic_to_array(self.all_results)
        matrix_all_results = np.asarray(matrix_all_results)
        df_all_results = self.displayResults(matrix_all_results)
        if train_model is False:
            self.all_results = []
            self.all_predictions = []
        return matrix_all_results, df_all_results, self.all_predictions


models = ['KNeighborsRegressor', 'BayesianRidge', 'DecisionTreeRegressor',
          'GradientBoostingRegressor', 'KernelRidgeRegression', 'GaussianProcessRegressor',
          'SuportVectorRegressor', 'MLPRegressor', 'AdaBoostRegressor']


def identify_index_model(model_to_identify):
    if model_to_identify is None:
        return 0
    for num_index, model in enumerate(models):
        if model == model_to_identify:
            return num_index


# =============================================================================
#                                   PLOTS
# =============================================================================

markers = ['-o', '-v', '-s', '-D', '-H', '-P', '-1', '-^', '-8']

values_to_compare = ['R2', 'Acc', 'MAPE', 'EVS', 'RMSE', 'MEDAE', 'MSLE']

models_used = ['K-Neighbors Regressor', 'Bayesian Ridge', 'Decision Tree Regressor',
               'Gradient Boosting Regressor', 'Kernel Ridge Regression', 'Gaussian Process Regressor',
               'Suport Vector Regressor', 'MLPRegressor', 'Ada Boost Regressor']

models_short = ['KNR', 'BR', 'DTR', 'GBR', 'KRR', 'GPR', 'SVR', 'MLPR', 'ABR']

eval_techniques = ['R2[%]', 'Accuracy[%]', 'MSE', 'RMSE', 'MAE', 'MSLE',
                   'MEDAE', 'MAPE[%]', 'EVS[%]', 'RMSE[%]', 'MEDAE[%]', 'MSLE[%]', 'Time[sec]', 'Mpred']

from sklearn.model_selection import learning_curve


class LearningCurve:
    def __init__(self, clf, X, Y, num_model, score, train_size, id_test, increased_ys=True):
        self.clf = clf
        self.X = X
        self.Y = Y
        self.num_model = num_model
        self.score = score
        self.train_size = train_size
        self.increased_ys = increased_ys
        self.id_test = id_test
    def __str__(self):
        return 'Model:{},score:{}'.format(self.clf, self.score)

    def save_figure(self, plt, name_plot):
        plt.savefig('graphics/' + name_plot + '_' + self.id_test + '.eps', format='eps', dpi=1000)

    @timer
    def plotLearningCurve(self):
        train_sizes, train_scores, test_scores = \
            learning_curve(self.clf, self.X, self.Y,
                           train_sizes=self.train_size,
                           scoring=self.score, cv=5, shuffle=True)  # !!cv=5

        train_scores_mean = -train_scores.mean(1)
        test_scores_mean = -test_scores.mean(1)
        if self.score == 'neg_mean_squared_error':
            gap = test_scores_mean - train_scores_mean
        else:
            gap = None

        plt.figure()
        # if self.increased_ys is False:
        #     plt.ylim([0, 5000])
        plt.plot(self.train_size, np.sqrt(test_scores_mean),
                 '-o', markerfacecolor="None", label=' test')
        plt.plot(self.train_size, np.sqrt(train_scores_mean),
                 '-v', markerfacecolor="None", label=' train')

        plt.xlabel("Train size")
        plt.ylabel('log('+ self.score+ ')')
        plt.title('Learning curve')
        plt.legend(loc="best")
        self.save_figure(plt, 'learningcurve')
        plt.show()
        return gap, train_sizes

    def plotGap(self, gap):
        plt.figure()
        plt.plot(self.train_size, gap, '-v', markerfacecolor="None", label='GAP Test-Train')
        plt.xlabel("Train size")
        plt.legend(loc="best")
        plt.ylabel('MSE')
        self.save_figure(plt, 'gap')
        plt.show()


def get_id_test(increased_ys, scale_data, input_pca=None, estimated=None, comment=None):
    if estimated:
        data_adq = 'estimated'
    elif estimated is None:
        data_adq = 'picture'
    else:
        data_adq = 'no_estimated'
    if increased_ys:
        output = 'inc'
    else:
        output = 'noinc'

    if scale_data:
        scaled = 'scaled'
    else:
        scaled = 'noscaled'
    if input_pca is True:
        data_adq = data_adq+'_pcainput'
    if comment:
        return comment + '_' + data_adq + '_' + output + '_' + scaled
    else:
        return data_adq + '_' + output + '_' + scaled


class Plots:

    def __init__(self, id_test, increased_ys=True):
        self.increased_ys = increased_ys
        self.id_test = id_test
        # plt.rcdefaults()

    def save_figure(self, plt, name_plot):
        plt.savefig('graphics/' + name_plot + '_' + self.id_test + '.eps', format='eps', dpi=1000)

    def heat_map(self, vector, title):
        plt.imshow(vector, cmap='hot', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar()
        cbar.set_label('Intensity [J/mm2]')
        plt.title("Heat Map Intensities")
        plt.savefig('graphics/' + title + '.eps', format='eps', dpi=1000)
        plt.show()

    def graph_model(self, vector, labels, model):
        plt.figure()
        plt.title(model)
        sc = plt.scatter(vector[:, 0], vector[:, 1], s=50, c=labels)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        self.save_figure(plt, model)
        plt.show()
        # plt.close()

    def ys_grainsize(self, X, Y):
        X_list_127, Y_list_127, X_list_84, Y_list_84, X_list_72, Y_list_72, X_list_35, Y_list_35 = split_energies(X, Y)
        x_s_list = [X_list_127, X_list_84, X_list_72, X_list_35]
        y_s_list = [Y_list_127, Y_list_84, Y_list_72, Y_list_35]

        plt.figure()
        for num_gamma, gamma_list in enumerate(x_s_list):
            z = np.polyfit(gamma_list[:, 2], y_s_list[num_gamma], 4)
            p = np.poly1d(z)
            plt.plot(gamma_list[:, 2], p(gamma_list[:, 2]), markers[num_gamma],
                     markerfacecolor="None", linewidth=1, markersize=6)
        plt.xlabel('d(nm)')
        if self.increased_ys:
            plt.ylabel('ΔYS[MPa]', fontsize=20)
        else:
            plt.ylabel('YS[MPa]', fontsize=20)
        plt.legend(['127', '84', '72', '35'], title='SFE', loc='best', fontsize=10)
        plt.rcParams.update({'font.size': 20})
        self.save_figure(plt, 'YSgrainsize')
        plt.show()

    def unsupervised_models(self, matrix, labels):
        labels = labels.astype(int)
        #### PCA ####
        pca = PCA()  # project from 64 to 2 dimensions
        pca_projected = pca.fit_transform(matrix)
        self.graph_model(pca_projected, labels, 'PCA')

        # print(pca.components_)
        print('Variance ratio matrix [%] with ', len(pca.explained_variance_ratio_), 'parameters:')
        variance_ratio_matrix = pca.explained_variance_ratio_ * 100
        print(variance_ratio_matrix.round())

        ##### ICA ####
        ica = ICA(n_components=2, random_state=None)
        ica_projected = ica.fit_transform(matrix)
        self.graph_model(ica_projected, labels, 'ICA')

        #### num components ####
        pca = PCA().fit(matrix)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        self.save_figure(plt, 'num_components')
        plt.show()

        #### tSNE ####
        tsne = TSNE(n_components=2, random_state=None,
                    perplexity=30)  # ULL amb la perplexity! Indica el num de veins a tenir en compte
        tsne_projected = tsne.fit_transform(matrix)
        self.graph_model(tsne_projected, labels, 'tSNE')

        #### LDA ####
        lda = LDA()  # ULL amb la perplexity! Indica el num de veins a tenir en compte
        lda_projected = lda.fit_transform(matrix, labels)
        self.graph_model(lda_projected, labels, 'LDA')

        ##### MDS ####
        mds = MDS(n_components=2, random_state=None)
        mds_projected = mds.fit_transform(matrix)
        self.graph_model(mds_projected, labels, 'MDS')

        ##### ALL ####
        fig, ax = plt.subplots(2, 2)
        sc1 = ax[0, 0].scatter(ica_projected[:, 0], ica_projected[:, 1], s=10, c=labels)  # row=0, col=0
        plt.colorbar(sc1, ax=ax[0, 0])
        ax[0, 0].set_title('ICA')
        sc2 = ax[1, 0].scatter(tsne_projected[:, 0], tsne_projected[:, 1], s=10, c=labels)  # row=1, col=0
        plt.colorbar(sc2, ax=ax[1, 0])
        ax[1, 0].set_title('tSNE')
        sc3 = ax[0, 1].scatter(lda_projected[:, 0], lda_projected[:, 1], s=10, c=labels)  # row=0, col=1
        plt.colorbar(sc3, ax=ax[0, 1])
        ax[0, 1].set_title('LDA')
        sc4 = ax[1, 1].scatter(mds_projected[:, 0], mds_projected[:, 1], s=10, c=labels)  # row=1, col=1
        plt.colorbar(sc4, ax=ax[1, 1])
        ax[1, 1].set_title('MDS')
        fig.tight_layout()
        self.save_figure(plt, 'UnsupervisedModels')
        plt.show()

    def pca_ncomponents(self, X, Y, num_components):
        plt.figure()
        pca = PCA(num_components)
        pca_projected = pca.fit_transform(X)
        ax = plt.subplot(111)
        ax.set_title('Feature 0-1', fontsize=14)
        sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        self.save_figure(plt, 'PCA_' + str(num_components))
        plt.show()

    def pca_three_components(self, X_scale, Y):
        plt.figure(figsize=(16, 4))
        pca = PCA(3)
        pca_projected = pca.fit_transform(X_scale)

        ax = plt.subplot(131)
        ax.set_title('Feature 0-1', fontsize=14)
        sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)


        ax = plt.subplot(132)
        ax.set_title('Feature 1-2', fontsize=14)
        sc = ax.scatter(pca_projected[:, 1], pca_projected[:, 2], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)

        ax = plt.subplot(133)
        ax.set_title('Feature 0-2', fontsize=14)
        sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 2], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        self.save_figure(plt, 'PCA_components')
        plt.show()

    def data_features(self, X, Y):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(copy.deepcopy(X))
        plt.figure(figsize=(16, 4))
        ax = plt.subplot(131)
        ax.set_xlabel('SFE [mJ/m$^2$]', fontsize=14)
        ax.set_ylabel('Std(SFE) [mJ/m$^2$]', fontsize=14)
        sc = ax.scatter(X_scale[:, 0], X_scale[:, 1], s=10, c=Y)
        clb = plt.colorbar(sc)
        clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        ax = plt.subplot(132)
        ax.set_xlabel('Std(SFE) [mJ/m$^2$]', fontsize=14)
        ax.set_ylabel('Region size [nm]', fontsize=14)
        sc = ax.scatter(X_scale[:, 1], X_scale[:, 2], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)

        ax = plt.subplot(133)
        ax.set_ylabel('Region size [nm]', fontsize=14)
        ax.set_xlabel('SFE [mJ/m$^2$]', fontsize=14)
        sc = ax.scatter(X_scale[:, 0], X_scale[:, 2], s=10, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        self.save_figure(plt, 'data_features')
        plt.show()

    def pca_components_3d(self, X_scale, Y):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)

        pca = PCA(3)
        pca_projected = pca.fit_transform(X_scale)

        sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2], s=50, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        self.save_figure(plt, 'pca_3d')
        plt.show()

    def evaluation_techniques(self, matrix_all_results):
        for num_value, row in enumerate(matrix_all_results[:, 7]):
            matrix_all_results[num_value, 7] = 100 - matrix_all_results[num_value, 7]

        values_percent = np.column_stack((matrix_all_results[:, :2], matrix_all_results[:, 7:12]))

        plt.figure()
        ax = plt.subplot(111)
        plt.xlabel('Evaluation technique')
        plt.ylabel('[%]')

        for num_row, row in enumerate(values_percent):
            #        sc=plt.scatter(values_to_compare,row,label=models_used[num_row],s=20)
            plt.plot(values_to_compare, row, markers[num_row], markerfacecolor="None", label=models_used[num_row])

        axes = plt.gca()
        # axes.set_xlim([xmin,xmax])
        axes.set_ylim([None, 100])
        # plt.colorbar(sc)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})

        plt.show()
        self.save_figure(plt, 'evaluation_techniques')
        plt.close()

    def ys_test_pred(self, all_predictions, testY, model_to_identify=None):
        plt.figure()
        plt.subplot(111)
        if self.increased_ys:
            plt.xlabel('ΔYS predicted [MPa]', fontsize=14)
            plt.ylabel('ΔYS actual [MPa]', fontsize=14)
        else:
            plt.xlabel('YS predicted [MPa]', fontsize=14)
            plt.ylabel('YS actual [MPa]', fontsize=14)

        if model_to_identify is None:
            index_model = 0
        else:
            index_model = identify_index_model(model_to_identify)
        plt.scatter(all_predictions[index_model], testY, s=20)

        all_predictions[index_model] = np.reshape(all_predictions[index_model], (len(all_predictions[index_model]),))
        z = np.polyfit(all_predictions[index_model], testY, 1)
        p = np.poly1d(z)
        plt.plot(all_predictions[index_model], p(all_predictions[index_model]), ':', linewidth=0.2)

        #    ax.legend([models_short[index_model]],loc='upper left', prop={'size': 12})
        plt.rcParams.update({'font.size': 14})
        plt.gca().set_aspect('equal', adjustable='box')
        axes = plt.gca()
        min_plot = min([min(all_predictions[index_model]), min(p(all_predictions[index_model]))])
        max_plot = max([max(all_predictions[index_model]), max(p(all_predictions[index_model]))])
        axes.set_xlim([min_plot - 0.2 * min_plot, max_plot + 0.1 * max_plot])
        axes.set_ylim([min_plot - 0.2 * min_plot, max_plot + 0.1 * max_plot])
        self.save_figure(plt, 'YStest_YSpred')
        plt.show()
        plt.close()

    def standard_deviation(self, all_predictions, testY, model_to_identify=None):
        plt.figure()
        ax = plt.subplot(111)
        plt.xlabel('Samples number')
        plt.ylabel('Standard deviation')

        index_model = identify_index_model(model_to_identify)
        vector_std = listStd(testY, all_predictions[index_model])

        objects = np.arange(len(vector_std))
        objects2 = []
        for number in objects:
            objects2 = np.append(objects2, str(number))
        y_pos = np.arange(len(objects))
        performance = vector_std

        plt.scatter(y_pos, performance, label=models_used[index_model], s=20)
        plt.plot(y_pos, performance, '-', linewidth=0.2)

        ax.legend([models_short[index_model]], loc='upper right', prop={'size': 12})
        plt.rcParams.update({'font.size': 14})
        self.save_figure(plt, 'std')
        plt.show()
        plt.close()

    def create_list_energies(self, prediction):
        list_35 = []
        list_72 = []
        list_84 = []
        list_127 = []
        for row in prediction:
            if row[3] == 35:
                list_35.append(row)
            elif row[3] == 72:
                list_72.append(row)
            elif row[3] == 84:
                list_84.append(row)
            elif row[3] == 127:
                list_127.append(row)
        return list_35, list_72, list_84, list_127

    def get_values_gamma(self, testX):
        number_35 = 0
        number_72 = 0
        number_84 = 0
        number_127 = 0
        for row in testX:
            if int(str(row[0])[:1]) == 1:
                number_127 += 1
            elif int(str(row[0])[:1]) == 3:
                number_35 += 1
            elif int(str(row[0])[:1]) == 8:
                number_84 += 1
            elif int(str(row[0])[:1]) == 7:
                number_72 += 1

        list_35 = np.full((number_35, 1), 35)
        list_72 = np.full((number_72, 1), 72)
        list84 = np.full((number_84, 1), 84)
        list_127 = np.full((number_127, 1), 127)
        values_gamma = np.concatenate((list_35, list_72, list84, list_127), axis=0)
        return values_gamma

    def ys_grainsize_prediction(self, testX, all_predictions):
        values_gamma = self.get_values_gamma(testX)

        for num_pred, prediction in enumerate(all_predictions):
            def getKey2(item):
                return item[1]

            prediction = np.column_stack((prediction, testX[:, 0]))
            prediction = np.column_stack((prediction, testX[:, 1]))
            prediction = sorted(prediction, key=getKey2)
            prediction = np.asarray(prediction)
            prediction = np.column_stack((prediction, values_gamma))

            list_35, list_72, list_84, list_127 = self.create_list_energies(prediction)

            def getKey(item):
                return item[2]

            list_35 = sorted(list_35, key=getKey)
            list_35 = np.asarray(list_35)

            list_72 = sorted(list_72, key=getKey)
            list_72 = np.asarray(list_72)

            list_84 = sorted(list_84, key=getKey)
            list_84 = np.asarray(list_84)

            list_127 = sorted(list_127, key=getKey)
            list_127 = np.asarray(list_127)

            marker_size = 6
            line_width = 1
            polynomial_grade = 3

            plt.figure()
            plt.title(models_used[num_pred])
            plt.xlabel('Region size [nm]')
            if self.increased_ys:
                plt.ylabel('ΔYS Predicted [MPa]')
            else:
                plt.ylabel('YS Predicted [MPa]')
            plt.scatter(list_35[:, 2], list_35[:, 0], s=0.1)
            z = np.polyfit(list_35[:, 2], list_35[:, 0], polynomial_grade)
            p = np.poly1d(z)
            plt.plot(list_35[:, 2], p(list_35[:, 2]), markers[0], markerfacecolor="None", linewidth=line_width,
                     markersize=marker_size)

            plt.scatter(list_72[:, 2], list_72[:, 0], s=0.1)
            z = np.polyfit(list_72[:, 2], list_72[:, 0], polynomial_grade)
            p = np.poly1d(z)
            plt.plot(list_72[:, 2], p(list_72[:, 2]), markers[1], markerfacecolor="None", linewidth=line_width,
                     markersize=marker_size)

            plt.scatter(list_84[:, 2], list_84[:, 0], s=0.1)
            z = np.polyfit(list_84[:, 2], list_84[:, 0], polynomial_grade)
            p = np.poly1d(z)
            plt.plot(list_84[:, 2], p(list_84[:, 2]), markers[2], markerfacecolor="None", linewidth=line_width,
                     markersize=marker_size)

            plt.scatter(list_127[:, 2], list_127[:, 0], s=0.1)
            z = np.polyfit(list_127[:, 2], list_127[:, 0], polynomial_grade)
            p = np.poly1d(z)
            plt.plot(list_127[:, 2], p(list_127[:, 2]), markers[3], markerfacecolor="None", linewidth=line_width,
                     markersize=marker_size)

            plt.legend(['35', '72', '84', '127'], loc='upper right', title='Gamma', prop={'size': 12})
            plt.rcParams.update({'font.size': 14})
            self.save_figure(plt, 'ys_grain')
            plt.show()
            plt.close()

    def gamma_grainsize(self, testX, predY):
        plt.figure()
        plt.xlabel('Region size [nm]')
        plt.ylabel('SFE[mJ/m$^2$')
        # plt.title('Standard deviaton for each prediction')
        sc = plt.scatter(testX[:, 1], testX[:, 0], s=50, c=predY[:, 0])
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        plt.rcParams.update({'font.size': 14})
        self.save_figure(plt, 'isf_grain')
        plt.show()
        plt.close()

    def std_grainsize(self, testX, testY, all_predictions, model_to_identify=None):
        index_model = identify_index_model(model_to_identify)
        vector_std = list_std_percent(testY, all_predictions[index_model])
        vector_std = [i * 100 for i in vector_std]

        objects = np.arange(len(vector_std))
        objects2 = []
        for number in objects:
            objects2 = np.append(objects2, str(number))
        performance = vector_std

        values_gamma = self.get_values_gamma(testX)

        def getKey2(item):
            return item[1]

        all_predictions[index_model] = np.column_stack((all_predictions[index_model], performance))

        all_predictions[index_model] = np.column_stack((all_predictions[index_model], testX[:, 1]))
        all_predictions[index_model] = sorted(all_predictions[index_model], key=getKey2)
        all_predictions[index_model] = np.asarray(all_predictions[index_model])
        all_predictions[index_model] = np.column_stack((all_predictions[index_model], values_gamma))

        list_35, list_72, list_84, list_127 = self.create_list_energies()

        def getKey(item):
            return item[2]

        list_35 = sorted(list_35, key=getKey)
        list_35 = np.asarray(list_35)

        list_72 = sorted(list_72, key=getKey)
        list_72 = np.asarray(list_72)

        list_84 = sorted(list_84, key=getKey)
        list_84 = np.asarray(list_84)

        list_127 = sorted(list_127, key=getKey)
        list_127 = np.asarray(list_127)

        marker_size = 6
        line_width = 1
        polynomial_grade = 3

        plt.figure()
        plt.title(model_to_identify)
        plt.xlabel('Region size [nm]')
        plt.ylabel('Stanard Deviation Normalized [%]')
        #    plt.scatter(list_35[:,2],list_35[:,1],s=0.1)
        z = np.polyfit(list_35[:, 2], list_35[:, 1], polynomial_grade)
        p = np.poly1d(z)
        plt.plot(list_35[:, 2], p(list_35[:, 2]), markers[0], markerfacecolor="None", linewidth=line_width,
                 markersize=marker_size)

        #    plt.scatter(list_72[:,2],list_72[:,1],s=0.1)
        z = np.polyfit(list_72[:, 2], list_72[:, 1], polynomial_grade)
        p = np.poly1d(z)
        plt.plot(list_72[:, 2], p(list_72[:, 2]), markers[1], markerfacecolor="None", linewidth=line_width,
                 markersize=marker_size)

        #    plt.scatter(list_84[:,2],list_84[:,1],s=0.1)
        z = np.polyfit(list_84[:, 2], list_84[:, 1], polynomial_grade)
        p = np.poly1d(z)
        plt.plot(list_84[:, 2], p(list_84[:, 2]), markers[2], markerfacecolor="None", linewidth=line_width,
                 markersize=marker_size)

        #    plt.scatter(list_127[:,2],list_127[:,1],s=0.1)
        z = np.polyfit(list_127[:, 2], list_127[:, 1], polynomial_grade)
        p = np.poly1d(z)
        plt.plot(list_127[:, 2], p(list_127[:, 2]), markers[3], markerfacecolor="None", linewidth=line_width,
                 markersize=marker_size)

        plt.legend(['35', '72', '84', '127'], loc='upper right', title='Gamma', prop={'size': 12})
        axes = plt.gca()
        axes.set_ylim([0, 5])
        plt.rcParams.update({'font.size': 14})
        self.save_figure(plt, 'std_grain')
        plt.show()
        plt.close()

    def invert_mape(self, matrix_all_results):
        for num_value, row in enumerate(matrix_all_results[:, 7]):
            matrix_all_results[num_value, 7] = 100 - matrix_all_results[num_value, 7]
        return matrix_all_results

    def evaltechniques_percentage(self, x_axis, matrix_all_results, x_label):
        values_to_compare = ['R2n', '1-MAPE', 'RMSEn']
        position = [0, 7, 9]
        matrix_all_results = self.invert_mape(copy.deepcopy(matrix_all_results))
        plt.figure()
        ax = plt.subplot(111)
        for num_evalTec, evaluation_tec in enumerate(values_to_compare):
            plt.plot(x_axis, matrix_all_results[:, position[num_evalTec]], markers[num_evalTec],
                     markerfacecolor="None", linewidth=0.5, label=evaluation_tec)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
        plt.legend(values_to_compare, loc='best', title='Metrics')
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel('[%]', fontsize=14)
        plt.rcParams.update({'font.size': 14})
        # axes = plt.gca()
        # axes.set_ylim([0, 100])
        self.save_figure(plt, 'evaltecPercent')
        plt.show()

    def evaltechniques_magnitude(self, Xaxis, matrix_all_results, X_label):
        matrix_all_results = self.invert_mape(matrix_all_results)
        fig = plt.figure()
        plt.xlabel(X_label, fontsize=14)
        ax1 = fig.add_subplot(111)
        ax1.plot(Xaxis, matrix_all_results[:, 0], '-o', markerfacecolor="None", linewidth=0.5, color='black',
                 label='R2')
        ax1.plot(Xaxis, matrix_all_results[:, 7], '-v', markerfacecolor="None", linewidth=0.5, color='black',
                 label='MAPE')
        ax1.plot(Xaxis, matrix_all_results[:, 8], '-s', markerfacecolor="None", linewidth=0.5, color='black',
                 label='EVS')
        ax1.set_ylabel('[%]', fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(Xaxis, matrix_all_results[:, 3], '-D', markerfacecolor="None", linewidth=0.5, color='red',
                 label='RMSE')
        ax2.plot(Xaxis, matrix_all_results[:, 4], '-H', markerfacecolor="None", linewidth=0.5, color='red', label='MAE')
        ax2.plot(Xaxis, matrix_all_results[:, 6], '-P', markerfacecolor="None", linewidth=0.5, color='red',
                 label='MEDAE')

        ax2.set_ylabel('', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.gca()
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1.1, 0.2), prop={'size': 12})

        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.8), prop={'size': 12})
        ax1.tick_params(axis='both', which='major', labelsize=14)
        plt.yticks(fontsize=14)
        self.save_figure(plt, 'evaltec')
        plt.show()

    def mape_one_axis(self, X, Y, mape, axis, label_axis):
        plt.figure()
        plt.ylabel('MAPE [%]', fontsize=14)
        plt.xlabel(label_axis, fontsize=14)
        sc = plt.scatter(X[:, axis], mape, s=50, c=Y)
        clb = plt.colorbar(sc)
        if self.increased_ys:
            clb.ax.set_title('ΔYS[MPa]', fontsize=14)
        else:
            clb.ax.set_title('YS[MPa]', fontsize=14)
        plt.rcParams.update({'font.size': 14})
        self.save_figure(plt, 'ysmape_'+label_axis[:3])
        plt.show()
        plt.close()

    def mape_all_features(self, X, Y, mape):
        all_labels = ['SFE[mJ/m$^2$]', 'Std(SFE)[mJ/m$^2$]', 'Region size[nm]']
        for num_axis, label in enumerate(all_labels):
            self.mape_one_axis(X, Y, mape, num_axis, label)

    # Data augmentation
    def add_scatter(self, lista, x, y, color):
        plt.scatter(x, y, c=color, s=10)
        lista = np.append(lista, [x, y])
        return lista

    def reshape_list(self, array):
        array = np.reshape(array[:2], (int(len(array[:2]) / 2), 2))
        return array

    def polyfit_aprox(self, x, y, marker, color, gamma):
        self.poly_func = np.polyfit(x, y, 3)
        p = np.poly1d(self.poly_func)
        print('Function of gamma {}:{}*x**3{}*x**2+{}*x+{}'.format(gamma,
                                                                   str(round(self.poly_func[0], 2)),
                                                                   str(int(self.poly_func[1])),
                                                                   str(int(self.poly_func[2])),
                                                                   str(int(self.poly_func[3]))))
        plt.plot(x, p(x), marker, markerfacecolor="None", linewidth=0.5, color=color)

    def add_noise(self, array, number):
        noise = np.random.normal(-5, 5, number)  # Noise of 5MPa
        array_noise = array + noise.astype(int)
        return array_noise

    def generate_values(self, gamma_list, number, gamma, random_gs):
        if random_gs is True:
            new_x = np.random.randint(1, 6, size=number)
        else:
            new_x = np.random.choice(np.unique(gamma_list[:, 0]), number)
        new_y = []
        for value in new_x:
            y_value = self.poly_func[0] * (value ** 3) + self.poly_func[1] * (value ** 2) + self.poly_func[2] * (
                value) + self.poly_func[3]
            new_y = np.append(new_y, y_value)
        new_y = self.add_noise(new_y, number)
        return new_x, new_y

    def scatter_newvalue(self, x, y, color):
        for num_value, x_value in enumerate(x):
            plt.scatter(x_value, y[num_value], edgecolors=color, s=90, marker="*", facecolors='none')

    def join_newvalues(self, list_x, list_y, new_samples):
        all_new_x = list_x[0]
        all_new_y = list_y[0]
        list_gamma = np.concatenate([[127] * new_samples,
                                     [84] * new_samples,
                                     [72] * new_samples,
                                     [35] * new_samples])

        list_std = np.concatenate([[39] * (new_samples * 3),
                                   [12] * new_samples])
        for num_gamma, gamma_x in enumerate(list_x):
            if num_gamma != 0:
                all_new_x = np.concatenate((all_new_x, gamma_x), axis=0)
                all_new_y = np.concatenate((all_new_y, list_y[num_gamma]), axis=0)

        x = np.vstack((list_gamma, list_std, all_new_x))
        x = np.transpose(x)
        y = all_new_y
        return x, y

    def data_augmentation_ys_grainsize(self, X, Y, new_samples, random_gs=True):
        color = ['red', 'black', 'green', 'purple']
        plt.figure()
        plt.ylabel('ΔYS [MPa]', fontsize=14)
        plt.xlabel('Region Size [nm]', fontsize=14)
        X_list_127, _, X_list_84, _, X_list_72, _, X_list_35, _ = split_energies(X, Y)

        _127_list = X_list_127[:, :2]
        self.polyfit_aprox(_127_list[:, 0], _127_list[:, 1], markers[0], color[0], '127')
        new127_x, new127_y = self.generate_values(_127_list, new_samples, '127', random_gs)
        self.scatter_newvalue(new127_x, new127_y, color[0])

        _84_list = X_list_84[:, :2]
        self.polyfit_aprox(_84_list[:, 0], _84_list[:, 1], markers[1], color[1], '84')
        new84_x, new84_y = self.generate_values(_84_list, new_samples, '84', random_gs)
        self.scatter_newvalue(new84_x, new84_y, color[1])

        _72_list = X_list_72[:, :2]
        self.polyfit_aprox(_72_list[:, 0], _72_list[:, 1], markers[2], color[2], '72')
        new72_x, new72_y = self.generate_values(_72_list, new_samples, '72', random_gs)
        self.scatter_newvalue(new72_x, new72_y, color[2])

        _35_list = X_list_35[:, :2]
        _35_list = np.asarray(sorted(_35_list, key=lambda x: x[0]))
        self.polyfit_aprox(_35_list[:, 0], _35_list[:, 1], markers[3], color[3], '35')
        new35_x, new35_y = self.generate_values(_35_list, new_samples, '35', random_gs)
        self.scatter_newvalue(new35_x, new35_y, color[3])

        plt.legend(['127', '84', '72', '35'], loc='upper right', title='Gamma', prop={'size': 10})
        self.save_figure(plt, 'dataugm')
        plt.show()
        plt.close()

        x_added, y_added = self.join_newvalues([new127_x, new84_x, new72_x, new35_x],
                                               [new127_y, new84_y, new72_y, new35_y],
                                               10)

        return x_added, y_added
