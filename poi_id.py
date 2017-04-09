#!/usr/bin/python

import sys
from pprint import pprint
from time import time
import pickle
from sklearn.preprocessing import Imputer, RobustScaler
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from ensemble import EnsembleClassifier
from sklearn import model_selection



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred',
                 'total_stock_value', 'loan_advances', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'deferred_income', 'expenses', 'from_poi_ratio']
                 ##'from_poi_to_this_person', 'to_messages']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### I used imputation for NaN and robust scaling for outliers

### Task 3: Create new feature(s)
### DONE: Add percentage of emails to poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

for person in my_dataset:
        if my_dataset[person]['from_poi_to_this_person'] != 'NaN' \
            and my_dataset[person]['to_messages'] != 'NaN':
            my_dataset[person]['from_poi_ratio'] = \
                my_dataset[person]['from_poi_to_this_person']/my_dataset[person]['to_messages']
        else:
            my_dataset[person]['from_poi_ratio'] = 0

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
robustScale = RobustScaler()
data = imputer.fit_transform(data)
data = robustScale.fit_transform(data)

targets, features = targetFeatureSplit(data)

features_train, features_test, targets_train, targets_test = \
    train_test_split(features, targets, test_size=0.2)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2, 3, 4]
             }

PCA = PCA(n_components=2)
#DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
svc = SVC()
ABC = AdaBoostClassifier(base_estimator=DTC)

clf = Pipeline(steps=[('pca', PCA),('dt', DTC), ('ada', ABC)])
print clf.get_params()


# run grid search
clf = GridSearchCV(ABC, param_grid=param_grid, scoring='roc_auc')
clf.fit(features_train, targets_train)
print 'accuracy', clf.score(features_test, targets_test)

pred = clf.predict(features_test)
actual = targets_test

precision = metrics.precision_score(actual, pred)
recall = metrics.recall_score(actual, pred)
print 'precision', precision
print 'recall', recall

'''
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid_search = GridSearchCV(ABC, param_grid=param_grid, scoring='roc_auc')




    grid_search.fit(features_train, targets_train)


    features_test, targets_pred = features_test, grid_search.predict(features_test)
    print targets_pred


# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

'''

dump_classifier_and_data(clf, my_dataset, features_list)