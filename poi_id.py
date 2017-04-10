import sys
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn import metrics
from sklearn import model_selection



### Task 1: Select what features you'll use.

### First, we load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Remove outliers
my_dataset = data_dict

def remove_key(dictionary, keys):
    dictionary.pop(keys, 0)

outliers_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for outlier in outliers_list:
    remove_key(my_dataset, outlier)

### Task 2: Make meaningful features and select features
finance_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

for person in my_dataset:
    my_dataset[person]['finAll'] = 0
    for f in finance_features:
        if f != 'salary':
            if my_dataset[person][f] != 'NaN' and my_dataset[person]['salary'] != 'NaN':
                my_dataset[person]['salary/'+f+'ratio'] = float(my_dataset[person]['salary'])/float(my_dataset[person][f])
            else:
                my_dataset[person]['salary/' + f + 'ratio'] = 'NaN'
            if my_dataset[person]['salary/'+f+'ratio'] != 'NaN':
                my_dataset[person]['finAll'] += float(my_dataset[person]['salary'])/float(my_dataset[person][f])
            else:
                my_dataset[person]['finAll'] += 0
            remove_key(my_dataset[person], f)
    remove_key(my_dataset[person], 'salary')

    if my_dataset[person]['from_poi_to_this_person'] != 'NaN' \
            and my_dataset[person]['from_messages'] != 'NaN':
        my_dataset[person]['from_poi_ratio'] = \
            my_dataset[person]['from_poi_to_this_person'] / my_dataset[person]['from_messages']
    else:
        my_dataset[person]['from_poi_ratio'] = 'NaN'

    for f in email_features:
        my_dataset[person]['emailAll'] = 0
        if f != 'from_messages' and f != 'email_address':
            if my_dataset[person][f] != 'NaN' and my_dataset[person]['from_messages'] != 'NaN':
                my_dataset[person][f+'/from_messages ratio'] = my_dataset[person][f]/(my_dataset[person]['from_messages'])
            else:
                my_dataset[person][f + '/from_messages ratio'] = 'NaN'

            if my_dataset[person][f+'/from_messages ratio'] != 'NaN':
                my_dataset[person]['emailAll'] += my_dataset[person][f+'/from_messages ratio']
            else:
                my_dataset[person]['emailAll'] += 0
            remove_key(my_dataset[person], f)
    remove_key(my_dataset[person], 'from_messages')
    remove_key(my_dataset[person], 'email_address')



all_keys = []

for person, k in my_dataset.items():
    for e in k.keys():
        if e not in all_keys:
            all_keys.append(e)

features_list = ['poi']

#for a in all_keys:
#    if a not in features_list:
#        features_list.append(a)
#print features_list
features_list = ['poi', 'finAll', 'emailAll', 'from_poi_ratio']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)

labels, features = targetFeatureSplit(data)
print features

#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.2)

#imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
#robustScale = MinMaxScaler(feature_range=(0, 1))
robustScale = Normalizer()
#data = imputer.fit_transform(data)
#data = robustScale.fit_transform(data)
#print data


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

pca = PCA()

param_grid = {#"ada__base_estimator__criterion" : ["gini", "entropy"],
              #"ada__base_estimator__splitter" :   ["best", "random"],
              #"ada__n_estimators": [50, 75, 100],
              'pca__n_components': [1, 2, 3],
              'dt__min_samples_split': [5, 10, 15, 20, 25, 35, 50],
              'dt__max_depth': [3, 4, 5, 6, 7],
              #'svc__kernel': ('linear', 'rbf', 'sigmoid'),
              #'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              #'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              #'kbest__k': [1, 2, 3, 4],
              #'kpercentile_percentile': [90],
            }

kbest = SelectKBest()
kpercentile = SelectPercentile(percentile=90)
svc = SVC()
DTC = DecisionTreeClassifier(max_features="auto", class_weight="balanced", max_depth=4)
nb = GaussianNB()
ABC = AdaBoostClassifier(base_estimator=DTC)

#pipe = Pipeline(steps=[('scale', robustScale), ('pca', pca), ('ada', ABC)])
pipe = Pipeline(steps=[('scale', robustScale), ('pca', pca), ('dt', DTC)])
#pipe = Pipeline(steps=[('scale', robustScale), ('pca', pca), ('nb', nb)])
#pipe = Pipeline(steps=[('scale', robustScale), ('svc', svc)])
#pipe = Pipeline(steps=[('scale', robustScale), ('pca', pca), ('kbest', kbest),('dt', DTC) ])
#pipe = Pipeline(steps=[('kpercentile', kpercentile), ('dt', DTC)])
#pipe = Pipeline(steps=[('scale', robustScale), ('kbest', kbest), ('ada', ABC)])

# run grid search
grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=sss)
grid_search.fit(features, labels)
clf = grid_search.best_estimator_

#print clf.score(features_test, labels_test)


#print labels_test
#print clf.predict(features_test)
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


dump_classifier_and_data(clf, my_dataset, features_list)

from tester import test_classifier
print ' '
# use test_classifier to evaluate the model
# selected by GridSearchCV
print "Tester Classification report"
test_classifier(clf, data_dict, features_list)

