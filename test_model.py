import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH="best_model.joblib"
#FIXTURE1:Load the trained model
@pytest.fixture
def model(): 
  assert os.path.exists(MODEL_PATH),"Model file not found!"
  return joblib.load(MODEL_PATH)
#FIXTURE 2:Prepare sample input data
@pytest.fixture
def sample_data():
  data=load_breast_cancer()
  x=pd.DataFrame(
  data.data,
  columns=data.feature_names
)
  return X.iloc[:5]
  #TEST 1:Check whether model loads successfully
  def test_model_loads(model):
     assert model is not None
    #TEST 2:Check whether model generates predictions
    def test_model_predction(model,sample_data)
    assert predctions is not None
#TEST 3:check prediction dimensions
def test_prediction_shape_shape(model,sample_data):
  predictions=model.predict(sample_data)
#TEST 4:Check valid prediction labe;s
def test_prediction_values(models,sample_data):
  predictions=model.predict(sample_data)
  assert set(predctions).issubset({0,1})
#TEST 5:Check predcition probabilities
def test_predction_probabilities(model,sample_data):
  probabilities=model.predict_proba(sample_data)
  assert probabilities.shape==(5,2)
  assert np.all(probabilities>=0)
  assert np.all(probabilities<=1)
  assert np.allclose(probabilities.sum(axis=1),1)
#TEST 6:Check pipeline structure
def test_model_pipeline(model):
 assert hasattr(model,"named_steps")
 assert "model" in model.named_steps
 assert "model" in model.named_steps
#TEST 7: Check reusable predcition function
def test_make_predction(model,sample_data):
 output=make_prediction(model,sample_data)
 assert "prediction" in output 
 assert output["predictioin'] in{"Malignant",Benign"}
#TEST 8:Check probabilities returned by prediction function
def test_make_prediction_probabilities(model,sample_data):
output=make_prediction(model,sample_data)
probabilities=output["probabilities"]
assert "malignant" in probabilities
assert "benign" in probabilities
assert 0 <= probabilities["malignant"]<=1
assert 0 <= probabilities["benign"]<=1
