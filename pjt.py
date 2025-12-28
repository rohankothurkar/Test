import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

st.title("Simple Linear Regression App")

x = st.number_input("Enter X value", value=1)
y_pred = model.predict([[x]])

st.write("Predicted Y value:", y_pred[0])
