import streamlit as st
import numpy as np
import joblib
from Model import Bio_marqueurs, Scoring_model
model = Scoring_model()
seuil_xgb= 0.22
#Interface
st.markdown('## Score of Paris')
Lymphocytes = st.number_input('Lymphocytes')
Basophiles = st.number_input('Basophiles')
Neutrophiles = st.number_input('Neutrophiles')
Eosinophiles = st.number_input('Eosinophiles')
Age = st.number_input('Age')
#print (Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)
def predict (Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles):
    prediction, probability = model.predict_covid(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)
    if probability [0][1]> seuil_xgb:
        return ("Positive to Covid19")
    else:
        return("Negative to Covid19")



#Predict button
if st.button('Predict'):
    #model = joblib.load('iris_model.pkl')
    #X = np.array([Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles])
    #st.markdown(predict_covid(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles))
    st.markdown(f'The prediction of Score of Paris based on the Xgboost model for this patient is : {predict(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)}')