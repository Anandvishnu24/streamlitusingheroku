# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:56:03 2024

@author: VISHNU
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:56:03 2024

@author: VISHNU
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
loaded_model = pickle.load(open('B:/ML/trained_model (2).sav', 'rb'))
scaler = pickle.load(open('B:/ML/scaler.sav', 'rb'))

# Streamlit title
st.title('Diabetes Prediction using Machine Learning')

# Input fields
Pregnancies = st.number_input('Number of Pregnancies', 0)
Glucose = st.number_input('Glucose Level', 0)
BloodPressure = st.number_input('Blood Pressure', 0)
SkinThickness = st.number_input('Skin Thickness', 0)
Insulin = st.number_input('Insulin Value', 0)
BMI = st.number_input('BMI Value', 0.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 0.0)
Age = st.number_input('Age', 0)

input_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
input_data_array = np.asarray(input_data)
input_reshape=input_data_array.reshape(1,-1)


sta_data = scaler.transform(input_data_array)
prediction = loaded_model.predict(sta_data)
if st.button('Diabetes Test Result'):
    if prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
    else:
        diab_diagnosis = 'The person is non-diabetic'
    
    st.success(diab_diagnosis)

    
                  