

import pandas as pd
import pickle


dataset = pd.read_csv('final1.csv')

X = dataset.iloc[:,[5,23,22,13,1,7,2,4,8,3]].values
y = dataset.iloc[:,24].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Dump the trained decision tree classifier with Pickle
svm_pkl_filename = 'svmclassifier.pkl'
# Open the file to save as pkl file
decision_tree_model_pkl = open(svm_pkl_filename, 'wb')
pickle.dump(classifier, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()