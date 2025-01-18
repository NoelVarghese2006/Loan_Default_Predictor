import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xbg

model = xbg.Booster()
model.load_model("xgb_model.bin")

X = [18, 8000, "OWN", 1.0, "PERSONAL", "B", 500.0, 10.0, 0.0625, "N", 1]

X = pd.DataFrame({
    'person_age': [18],
    'person_income': [8000],
    'person_home_ownership': ['OWN'],
    'person_emp_length': [1.0],
    'loan_intent': ['PERSONAL'],
    'loan_grade': ['B'],
    'loan_amnt': [500.0],
    'loan_int_rate': [10.0],
    'loan_percent_income': [0.0625],
    'cb_person_default_on_file': ['N'],
    'cb_person_cred_hist_length': [1]
})


cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')

to_predict = xbg.DMatrix(X, enable_categorical=True)

default = model.predict(to_predict)

print(model.predict(to_predict))

if(default[0] < .5):
   print("This loan most probably won't default")
else:
   print("This loan will most likely default")