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
