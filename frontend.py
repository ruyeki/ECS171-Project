import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
import pickle

try:
    with open('svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)
except FileNotFoundError:
    print("File not found. Make sure the path to the pickled model is correct.")

#the prediction using svm model
def predict(input_data, svm_model):
    prediction = svm_model.predict(input_data)
    return prediction

# Streamlit app part of code
def main():
    #load the model into program
    with open('svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)

    st.title('Using SVM model to predict if a business will go bankrupt')

    # Input fields (still stuck on this part)
    input_field1 = st.slider('Input Feature 1', min_value=0.0, max_value=1.0)

    input_field2 = st.slider('Input Feature 2', min_value=0.0, max_value=1.0)

    input_field3 = st.slider('Input Feature 3', min_value=0.0, max_value=1.0)

    input_field4 = st.slider('Input Feature 4', min_value=0.0, max_value=1.0)

    input_field5 = st.slider('Input Feature 5', min_value=0.0, max_value=1.0)

    input_field6 = st.slider('Input Feature 6', min_value=0.0, max_value=1.0)

    input_field7 = st.slider('Input Feature 7', min_value=0.0, max_value=1.0)

    input_field8 = st.slider('Input Feature 8', min_value=0.0, max_value=1.0)

    input_field9 = st.slider('Input Feature 9', min_value=0.0, max_value=1.0)

    input_field10 = st.slider('Input Feature 10', min_value=0.0, max_value=1.0)

    input_field11 = st.slider('Input Feature 11', min_value=0.0, max_value=1.0)

    input_field12 = st.slider('Input Feature 12', min_value=0.0, max_value=1.0)

    input_field13 = st.slider('Input Feature 13', min_value=0.0, max_value=1.0)

    input_field14 = st.slider('Input Feature 14', min_value=0.0, max_value=1.0)

    input_field15 = st.slider('Input Feature 15', min_value=0.0, max_value=1.0)

    input_field16 = st.slider('Input Feature 16', min_value=0.0, max_value=1.0)

    input_field17 = st.slider('Input Feature 17', min_value=0.0, max_value=1.0)

    input_field18 = st.slider('Input Feature 18', min_value=0.0, max_value=1.0)

    input_field19 = st.slider('Input Feature 19', min_value=0.0, max_value=1.0)

    input_field20 = st.slider('Input Feature 20', min_value=0.0, max_value=1.0)

    input_field21 = st.slider('Input Feature 21', min_value=0.0, max_value=1.0)

    input_data = np.array([[input_field1, input_field2, input_field3, input_field4,input_field5, input_field6,input_field7, input_field8,input_field9, input_field10,input_field11, input_field12,input_field13, input_field14,input_field15, input_field16,input_field17, input_field18,input_field19, input_field20,input_field21]])  # Prepare input data for prediction
    
    #Make prediction once you hit the button
    if st.button('Predict'):
        prediction = predict(input_data, svm_model)
        if prediction == 1:
            st.write("The company will go bankrupt")
        elif prediction == 0:
            st.write("The company will not go bankrupt")
        else:
            st.write("Error with inputs")


if __name__ == '__main__':
    main()
