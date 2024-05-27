import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('cancer patient data sets.csv')

level_mapping = {'Low': float(0), 'Medium': float(1.0), 'High': float(2.0)}

df['Level'] = df['Level'].replace(level_mapping)

X = df.iloc[:, 2:-1].values  # Select all rows and columns from index 2 (excluding Level and index, Patient Id) up to the last column
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create a SVC
from sklearn import svm
model = svm.SVC(kernel="linear")

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
print(y_pred)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', round(accuracy*100, 2), '%')


# Save the model
filename = 'svc_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model, f)


# Save the scaler object to a file
filename = 'scaler.pkl'
with open(filename, 'wb') as f:
    pickle.dump(scaler, f)
