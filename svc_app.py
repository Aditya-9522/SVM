#create a streamlit application for the svc.ipynb referencer
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
st.title("Support Vector Machine (SVM) Classifier")
st.write("This application allows you to train a Support Vector Machine (SVM) classifier on the Iris dataset.")
# Select features and target variable
features = st.multiselect("Select features for training:", df.columns[:-1].tolist(), default=df.columns[:-1].tolist())
target = st.selectbox("Select target variable:", df.columns[-1])
# Train the SVM classifier
if st.button("Train SVM Classifier"):
    X = df[features]
    y = df[target]
    model = svm.SVC(kernel='linear')
    model.fit(X, y)
    st.success("SVM Classifier trained successfully!")
    # Display the coefficients of the trained model
    st.write("Coefficients of the trained SVM model:")
    st.write(model.coef_)
    st.write("Intercept of the trained SVM model:")
    st.write(model.intercept_)
    # Display the support vectors
    st.write("Support vectors of the trained SVM model:")
    st.write(model.support_vectors_)
    