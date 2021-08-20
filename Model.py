
# 1. Library imports
import numpy
import pandas as pd 
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler


# 2. Class which describes features
class Bio_marqueurs(BaseModel):
    Age: int
    Lymphocytes: float 
    Basophiles: float 
    Neutrophiles: float 
    Eosinophiles: float

# 3. Class for loading the model and making predictions
class Scoring_model:
    def __init__(self):
        self.model_fname_ = 'model_boosting.pkl'
        self.model = joblib.load(self.model_fname_)
        self.cr = joblib.load('StandarScaler.pkl') 
    #    Make a prediction based on the user-entered data
    #    Returns the prediction with its respective probability
    def predict_covid( self,Age,Lymphocytes, Basophiles, Neutrophiles, Eosinophiles):
        data=[{'Age':Age,'Lymphocytes':Lymphocytes,'Basophiles':Basophiles,'Neutrophiles':Neutrophiles,'Eosinophiles':Eosinophiles}]
        data_in=pd.DataFrame(data,index=['patient'])
        data_in=self.cr.transform(data_in)
        prediction = self.model.predict(data_in).tolist()
        #print(prediction)
        #print(type(prediction))
        probability = self.model.predict_proba(data_in).tolist()
        #print(probability)
        #print(type(probability))

        return prediction, probability
    
        
