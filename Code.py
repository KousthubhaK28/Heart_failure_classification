# Step-1: Upload your Kaggle API key file here
from google.colab import files
files.upload()

# Step 2: Data Preprocessing
import pandas as pd

# Load the dataset
data = pd.read_csv('heart.csv')

# Display the first few rows
data.head()

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Normalize or scale numerical features if necessary
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('HeartDisease', axis=1))

# Convert the scaled data back to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=data.columns[:-1])
scaled_data['HeartDisease'] = data['HeartDisease']

# Step 3: Feature Selection/Engineering
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the correlation matrix
corr = scaled_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Feature selection based on correlation or other techniques
X = scaled_data.drop('HeartDisease', axis=1)
y = scaled_data['HeartDisease']

# Step 4: Data Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Select and Train a Model
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

# Step 6: Visualize the Results
import matplotlib.pyplot as plt

# Plotting accuracies
models = ['Logistic Regression', 'Random Forest', 'SVC']
accuracies = [accuracy_score(y_test, log_reg_pred), accuracy_score(y_test, rf_clf_pred), accuracy_score(y_test, svc_clf_pred)]

plt.figure(figsize=(10, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()
