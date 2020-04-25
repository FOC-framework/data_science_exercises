# useful links
# link to datasets: https://vincentarelbundock.github.io/Rdatasets/datasets.html
# From this link: https://www.datacamp.com/community/tutorials/introduction-factor-analysis

from matplotlib.pyplot import plot

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import FactorAnalyzer

from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

do_plot = True
verbose = True

# function to plot in 2D the 2 variables dimensions
# this can be used for PCA as well as the Factor Analysis
# this is why we pass as parameter the label prefix
def plotIn2D(X_train, component_label_prefix, plotIndex):
    principalDf = pd.DataFrame(data=X_train, columns=[component_label_prefix + ' 1', component_label_prefix + ' 2'])
    print(X_train_pca)
    print("Explained Variance")
    print("------------------")
    print(pca.explained_variance_)

    if do_plot:
        plt.figure(plotIndex, figsize=(10, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel(component_label_prefix + ' - 1', fontsize=20)
        plt.ylabel(component_label_prefix + ' - 2', fontsize=20)
        plt.title(component_label_prefix + " Analysis of Breast Cancer Dataset", fontsize=20)
        targetsLabels = ['Benign', 'Malignant']
        targetsIndices = [1, 0]
        colors = ['r', 'g']
        for target, color in zip(targetsIndices, colors):
            indicesToKeep = y_train == target
            plt.scatter(principalDf.loc[indicesToKeep, component_label_prefix + ' 1'],
                        principalDf.loc[indicesToKeep, component_label_prefix + ' 2'], c=color, s=50)

        plt.legend(targetsLabels, prop={'size': 15})


# print the data set keys to help explore
def print_data_set_keys(data_set):
    # displaying the different keys of the data set and their contents
    print('keys')
    print('----')
    print(data_set.keys())


# also for exploration purposes this prints the values of every key
def print_values_for_every_key(data_set):
    for key in data_set.keys():
        print(key)
        print('-------------------')
        print(data_set[key])


# loading the data set
data_set = sklearn.datasets.load_breast_cancer()

if verbose:
    print_data_set_keys(data_set)
    print_values_for_every_key(data_set)

# original features and targets
features = data_set['data']
targets = data_set['target']

# target is an array of 0 / 1
if verbose:
    the_sum = sum(targets)
    print(the_sum)  # = 357 and there are 357 Benign => 1 means Benign

# split the data in train and test. It is important to have a specified random state so that
# the split become deterministic and repeats itself, allowing comparison of different runs.
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=42)

# scaler object allows to scale all features.
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)  # scaler fits the train but will obviously be applied to both train and test features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# from now on we will only use the scaled features

# print('Scaler mean array :' + str(scaler.mean_))

# The plan
# Explore 1: PCA with 20 variables in order to find the ideal dimension
# Explore 2: 2D plot of all individuals using the 2D PCA vs 2D Factor analysis
# Method 1: Logistic regression => confusion matrix LR
# Method 2: PCA 4D -> then logistic regression => confusion matrix PCA-LR
# Method 3: Factor Analysis 4D -> then logistic regression => confusion matrix FA-LR
# Conclusion:

# ---------------------------------------------------------------------
# Explore 1: PCA with 20 variables in order to find the ideal dimension
# ---------------------------------------------------------------------
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X_train_scaled)

if verbose:
    print(principalComponents)
    print(pca.mean_)

    print("Explained Variance")
    print("------------------")
    print(pca.explained_variance_)

count_axis = range(1, 21)

# Drawing this plot allows to decide on the number of variables to be taken
# the graph would show that 4 is acceptable
if do_plot:
    sn.scatterplot(count_axis, pca.explained_variance_)
    plt.show()

# ----------------------------------------------------------------------------
# Explore 2: 2D plot of all individuals using the 2D PCA vs 2D Factor analysis
# ----------------------------------------------------------------------------

# PCA 2d
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
plotIn2D(X_train_pca, 'principal component', 1)

# FA 2d
fa = FactorAnalyzer(rotation=None, n_factors=2)
fa.fit(X_train_scaled)
X_train_fa = fa.transform(X_train_scaled)
plotIn2D(X_train_fa, 'Factor analysis', 2)

# the following instruction shows the 2 graphs in 2D PCA and FA.
# we notice that the graphs are very similar in distribution of the individuals
# The plan allows a clear separation of Benin from Malign
plt.show()

# X_test_fa = fa.transform(X_test_scaled)
# print("X_train_fa")
# print(X_train_fa)

# Method 1: Logistic regression
# -----------------------------
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_scaled, y_train)
y_pred = logistic_regression.predict(X_test_scaled)
# print(y_pred)
# print(y_test)

# Confusion matrix to judge the efficiency of logistic regression
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
if do_plot:
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

# Method 2: PCA 4D -> then logistic regression => confusion matrix PCA-LR
# -----------------------------------------------------------------------
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.fit_transform(X_test_scaled)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_pca, y_train)
y_pred = logistic_regression.predict(X_test_pca)

# Confusion matrix to judge the efficiency of logistic regression
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
if do_plot:
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

# Method 3: Factor Analysis 4D -> then logistic regression => confusion matrix FA-LR
# ----------------------------------------------------------------------------------
fa = FactorAnalyzer(rotation=None, n_factors=4)
fa.fit(X_train_scaled)
X_train_fa = fa.transform(X_train_scaled)
X_test_fa = fa.transform(X_test_scaled)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_fa, y_train)
y_pred = logistic_regression.predict(X_test_fa)

# Confusion matrix to judge the efficiency of logistic regression
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
if do_plot:
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

# -----------
# Conclusion:
# -----------
'''
As expected using PCA or Factor Analysis variable reductions reduced the accuracy of the Logistic regression output
Yet results remained accurate enough. In larger scale data these methods would have dramatically reduced the training 
time, keeping an acceptable accuracy.
'''





# DRAFT
# -----

'''

# Kaiser-Meyer-Olkin (KMO) Test:
# Measuring the adequacy of a data set for Factor Analysis
from factor_analyzer.factor_analyzer import calculate_kmo

kmo_all, kmo_model = calculate_kmo(X_train_scaled)
print(kmo_model)
# Since the KMO result 0.8 > 0.6 => Data adequate for factor analysis

# CHI 2 matrix
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

chi_square_value, p_value = calculate_bartlett_sphericity(X_train_scaled)
print(chi_square_value)
print(p_value)

# Getting eigenvalues
# from factor_analyzer.factor_analyzer import FactorAnalyzer
# fa = FactorAnalyzer(X_train_scaled)
# ev = faor_analyzer.factor_analyzer import FactorAnalyzer

'''