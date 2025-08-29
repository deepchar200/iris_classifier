import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("A simple ML web app built with Streamlit")

# User inputs
sepal_length = st.slider("Sepal length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
sepal_width = st.slider("Sepal width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
petal_length = st.slider("Petal length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
petal_width = st.slider("Petal width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

# Make prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_class = iris.target_names[prediction][0]

st.subheader("🌼 Prediction:")
st.write(f"The predicted flower is **{predicted_class}**")