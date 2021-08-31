import streamlit as st
import numpy as np
import joblib
from Model import Bio_marqueurs, Scoring_model
model = Scoring_model()
seuil_xgb= 0.22
#Interface
st.markdown('## Score of Paris')
Lymphocytes = st.number_input('Lymphocytes (G/L)')
Basophiles = st.number_input('Basophiles (G/L)')
Neutrophiles = st.number_input('Neutrophiles (G/L)')
Eosinophiles = st.number_input('Eosinophiles (G/L)')
Age = st.number_input('Age (years)')
#print (Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)
def predict (Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles):
    prediction, probability = model.predict_covid(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)
    prob=  probability [0][1]
    if prob> seuil_xgb:
            return (f'Positive to Covid19 with a probability equal to '+'%0.2f' %prob)
    else:
        return(f'Negative to Covid19 with a probability equal to '+'%0.2f' %prob)



#Predict button
if st.button('Predict'):
    #model = joblib.load('iris_model.pkl')
    #X = np.array([Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles])
    #st.markdown(predict_covid(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles))
    st.markdown(f'The prediction of Score of Paris based on the Xgboost model for this patient is : {predict(Age, Lymphocytes, Basophiles, Neutrophiles,Eosinophiles)}.')
    st.markdown(f'The threshold of the xgboost model which gives a sensibility of 90% is equal to 0.22 .')
