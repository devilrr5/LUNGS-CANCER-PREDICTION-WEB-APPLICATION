import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.tree import DecisionTreeClassifier
from flask import Flask
app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# Load the scaler object used during training
scaler = StandardScaler()
scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# Load the data for prediction as a numpy array
my_list0 = [33,1,2,4,5,4,3,2,2,4,3,2,2,4,3,4,2,2,3,1,2,3,4]
my_list1 = [17,1,3,1,5,3,4,2,2,2,2,4,2,3,1,3,7,8,6,2,1,7,2]
my_list2 = [64,2,6,8,7,7,7,6,7,7,7,8,7,7,9,6,5,7,2,4,3,1,4]

# Convert the list to a numpy array with the desired shape
new_data = np.array([my_list2])

# Standardize the new data using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions
predicted_classes = model.predict(new_data_scaled)

# Print the predicted class
print("Predicted class:", predicted_classes[0], type(predicted_classes[0]))
