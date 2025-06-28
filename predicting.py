import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xbg

def predict(age, income, home, employment, intent, grade, amt, int_rate, pct_inc, def_hist, hist_len):
   model = xbg.Booster()
   model.load_model("xgb_model.bin")

   X = [18, 8000, "OWN", 1.0, "PERSONAL", "B", 500.0, 10.0, 0.0625, "N", 1]

   X = pd.DataFrame({
      'person_age': [age],
      'person_income': [income],
      'person_home_ownership': [home],
      'person_emp_length': [employment],
      'loan_intent': [intent],
      'loan_grade': [grade],
      'loan_amnt': [amt],
      'loan_int_rate': [int_rate],
      'loan_percent_income': [pct_inc],
      'cb_person_default_on_file': [def_hist],
      'cb_person_cred_hist_length': [hist_len]
   })


   cats = X.select_dtypes(exclude=np.number).columns.tolist()

   # Convert to Pandas category
   for col in cats:
      X[col] = X[col].astype('category')

   to_predict = xbg.DMatrix(X, enable_categorical=True)

   default = model.predict(to_predict)

   print(model.predict(to_predict))

   if(default[0] < .5):
      return("This loan most probably won't default")
   else:
      return("This loan will most likely default")