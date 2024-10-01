import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

# Load the dataset
data = pd.read_csv("HR_comma_sep.csv")

data=data.drop(columns=["sales"])
# Title of the app
st.title("HR Employee Attrition Analysis")

# Display the dataset
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# Visualize the distribution of employees who left
st.subheader("Distribution of Employees Who Left")
left_count = data.groupby('left').count()
fig1, ax1 = plt.subplots()
ax1.bar(left_count.index.values, left_count['satisfaction_level'])
ax1.set_xlabel('Employees Left Company')
ax1.set_ylabel('Number of Employees')
ax1.set_title('Distribution of Employees Who Left')
st.pyplot(fig1)

# Visualize the number of employees by number of projects
st.subheader("Number of Employees by Number of Projects")
num_projects = data.groupby('number_project').count()
fig2, ax2 = plt.subplots()
ax2.bar(num_projects.index.values, num_projects['satisfaction_level'])
ax2.set_xlabel('Number of Projects')
ax2.set_ylabel('Number of Employees')
st.pyplot(fig2)

# Visualize the number of employees by years spent in the company
st.subheader("Number of Employees by Years Spent in Company")
time_spent = data.groupby('time_spend_company').count()
fig3, ax3 = plt.subplots()
ax3.bar(time_spent.index.values, time_spent['satisfaction_level'])
ax3.set_xlabel('Number of Years Spent in Company')
ax3.set_ylabel('Number of Employees')
st.pyplot(fig3)

# Prepare the features and target variable
X = data.drop(columns=["left"])
y = data["left"]


# Encode the salary column
le = LabelEncoder()
X["salary"] = le.fit_transform(X["salary"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the neural network
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit the model
ann.fit(X_train, y_train, epochs=10)

# Making predictions
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Prepare for comparison
y_test_np = y_test.to_numpy()
result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test_np.reshape(len(y_test_np), 1)), axis=1)

# Display the results
st.subheader("Predictions vs Actual")
st.write(result)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
st.write(cm)

accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

# User input for prediction

st.subheader("Predict Employee Attrition")
user_input = st.text_input("Enter values for prediction (number_project, time_spend_company, Work_accident, promotion_last_5years, Departments, salary) separated by commas:")
if user_input:
    input_data = [float(x) for x in user_input.split(',')]
    input_scaled = sc.transform([input_data])
    prediction = ann.predict(input_scaled)
    st.write("Prediction (Left or Not Left):", prediction > 0.5)


