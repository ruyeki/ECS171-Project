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
    return prediction[0]

# Streamlit app part of code
def main():
    #load the model into program
    with open('svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)

    st.title('Predicting Business Bankrupcy')
    st.write('Input data using the sliders to see if a business will go bankrupt.')

    # Input fields (idk if this part is correct?)
    input_field1 = st.slider('ROA(C) before interest and depreciation before interest', min_value=0.0, max_value=1.0)

    input_field2 = st.slider('ROA(A) before interest and after tax', min_value=0.0, max_value=1.0)

    input_field3 = st.slider('ROA(B) before interest and depreciation after tax', min_value=0.0, max_value=1.0)

    input_field4 = st.slider('Net Value Per Share (B)', min_value=0.0, max_value=1.0)

    input_field5 = st.slider('Net Value Per Share (A)', min_value=0.0, max_value=1.0)

    input_field6 = st.slider('Net Value Per Share (C)', min_value=0.0, max_value=1.0)

    input_field7 = st.slider('Persistent EPS in the Last Four Seasons', min_value=0.0, max_value=1.0)

    input_field8 = st.slider('Per Share Net profit before tax (Yuan ¥)', min_value=0.0, max_value=1.0)

    input_field9 = st.slider('Debt ratio %', min_value=0.0, max_value=1.0)

    input_field10 = st.slider('Net worth/Assets', min_value=0.0, max_value=1.0)

    input_field11 = st.slider('Borrowing dependency', min_value=0.0, max_value=1.0)

    input_field12 = st.slider('Net profit before tax/Paid-in capital', min_value=0.0, max_value=1.0)

    input_field13 = st.slider('Working Capital to Total Assets', min_value=0.0, max_value=1.0)

    input_field14 = st.slider('Current Liability to Assets', min_value=0.0, max_value=1.0)

    input_field15 = st.slider('Current Liabilities/Equity', min_value=0.0, max_value=1.0)

    input_field16 = st.slider('Retained Earnings to Total Assets', min_value=0.0, max_value=1.0)

    input_field17 = st.slider('Current Liability to Equity', min_value=0.0, max_value=1.0)

    input_field18 = st.slider('Current Liability to Current Assets', min_value=0.0, max_value=1.0)

    input_field19 = st.slider('Net Income to Total Assets', min_value=0.0, max_value=1.0)

    input_field20 = st.slider('Net Income to Stockholders Equity', min_value=0.0, max_value=1.0)

    input_field21 = st.slider('Liability to Equity', min_value=0.0, max_value=1.0)

    input_data = np.array([[input_field1, input_field2, input_field3, input_field4,input_field5, input_field6,input_field7, input_field8,input_field9, input_field10,input_field11, input_field12,input_field13, input_field14,input_field15, input_field16,input_field17, input_field18,input_field19, input_field20,input_field21]])  # Prepare input data for prediction
    
    st.write(input_data)
    
    #Make prediction once you hit the button
    if st.button('Prediction'):
        
        prediction = predict(input_data, svm_model)
        st.write("Prediction shape:", prediction.shape)
        
        if prediction == 1:
            
            #1 represents positive class, bankrupt
            st.write("The company will go bankrupt")
            
        elif prediction == 0:
            
            #0 represents negative class, not bankrupt
            st.write("The company will not go bankrupt")
            
        else:
            st.write("An error has occured")
        
        st.write(prediction)
    


if __name__ == '__main__':
    main()
