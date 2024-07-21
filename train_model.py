from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Initialize models
log_reg = LogisticRegression()
rf_clf = RandomForestClassifier()
svc_clf = SVC()

# Train the models
log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
svc_clf.fit(X_train, y_train)

# Predict using the models
log_reg_pred = log_reg.predict(X_test)
rf_clf_pred = rf_clf.predict(X_test)
svc_clf_pred = svc_clf.predict(X_test)

# Evaluate the models
print("Logistic Regression:")
print(classification_report(y_test, log_reg_pred))
print(f"Accuracy: {accuracy_score(y_test, log_reg_pred)}\n")

print("Random Forest Classifier:")
print(classification_report(y_test, rf_clf_pred))
print(f"Accuracy: {accuracy_score(y_test, rf_clf_pred)}\n")

print("Support Vector Classifier:")
print(classification_report(y_test, svc_clf_pred))
print(f"Accuracy: {accuracy_score(y_test, svc_clf_pred)}\n")
