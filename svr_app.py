#create a streamlit application for the svr.ipynb referencer with iris dataset
import streamlit as st
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
# Load the iris dataset
iris = load_iris()
# Create a DataFrame from the iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target variable to the DataFrame
df['target'] = iris.target
# Create a Streamlit application
st.title("Support Vector Regression (SVR) Regressor")
st.write("This application allows you to train a Support Vector Regression (SVR) regressor on the Iris dataset.")
# Select features and target variable
features = st.multiselect("Select features for training:", df.columns[:-1].tolist(), default=df.columns[:-1].tolist())
target = st.selectbox("Select target variable:", df.columns[-1])
# Train the SVR regressor
if st.button("Train SVR Regressor"):
    X = df[features]
    y = df[target]
    model = svm.SVR(kernel='linear')
    model.fit(X, y)
    st.success("SVR Regressor trained successfully!")
    # Display the coefficients of the trained model
    st.write("Coefficients of the trained SVR model:")
    st.write(model.coef_)
    st.write("Intercept of the trained SVR model:")
    st.write(model.intercept_)
