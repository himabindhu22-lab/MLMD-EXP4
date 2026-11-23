#predict.py

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

#STEP1: Load the saved model
def load_mode(model_path="best_model.joblib"):
  model=joblib.load(model_path)
  return model
#STEP2: create a reusable prediction function
def make_prediction(model,input_data):
  prediction=model.predict(input_data):
  probabilities=model.predict_proba(input_data)
  result="Malignant" if prediction[0]==0 else"Begin"
  return{
   "prediction":result,
   "probabilities":{
     "malignant":probabilities[0][0],
     "begnign":probabilities[0][1]
   }
  }
#STEP 3:Run a sample prediction when executed directly
if __name__=="__main__":
  model=load_model()
  data=load_breast_cancer()
  feature_names=data.feature_namws
  sample_input["mean radius"]=15.0
  sample_input["mean texture"]=18.0
  sample_input["mean perimeter"]=100.0
  input-df=pd.DataFrame(
    [sample_input],
  columns=feature_names
  )
  output=make_prediction(model,input_df)
  print("Model loaded successfully!")
  print("\nPrediction:",output["prediction"])
  print("\nPrediction Probabilities:")
  print("Maligant:",output["probabilities"]["maligant"])
  print("Benign:",output["probabilities"]["benign"])
  
