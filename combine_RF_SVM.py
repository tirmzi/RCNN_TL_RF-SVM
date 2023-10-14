import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


features = np.load('featuresDCNN.npy')
labels = np.load('labelsObtained.npy')


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)

combined_predictions = np.bitwise_or(rf_predictions, svm_predictions)


confusion = confusion_matrix(y_test, combined_predictions)
classification_rep = classification_report(y_test, combined_predictions)

#  Displaying various Confusion Matrix, Accuracy and Classification reports....
print("Confusion Matrix:\n", confusion)