#create a streamlit application for the svr.ipynb referencer
import streamlit as st
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_boston
# Load the Boston housing dataset
boston = load_boston()
# Create a DataFrame from the Boston housing dataset
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
# Add the target variable to the DataFrame
df['target'] = boston.target
# Create a Streamlit application
st.title("Support Vector Regression (SVR)")
st.write("This application allows you to train a Support Vector Regression (SVR) model on the Boston housing dataset.")
# Select features and target variable
features = st.multiselect("Select features for training:", df.columns[:-1].tolist(), default=df.columns[:-1].tolist())
target = st.selectbox("Select target variable:", df.columns[-1])
# Train the SVR model
if st.button("Train SVR Model"):
    X = df[features]
    y = df[target]
    model = svm.SVR(kernel='linear')
    model.fit(X, y)
    st.success("SVR Model trained successfully!")
    # Display the coefficients of the trained model
    st.write("Coefficients of the trained SVR model:")
    st.write(model.coef_)
    st.write("Intercept of the trained SVR model:")
    st.write(model.intercept_)