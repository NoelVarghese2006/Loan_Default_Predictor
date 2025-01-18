import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xbg

#warnings.filterwarnings("ignore")

loan_data = pd.read_csv("credit_risk_dataset.csv")

#print(loan_data.describe(exclude=np.number))

#print(loan_data['loan_status'].value_counts())

X, y = loan_data.drop('loan_status', axis=1), loan_data[['loan_status']]

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')

#print(X.dtypes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

train_set = xbg.DMatrix(X_train, y_train, enable_categorical=True)
test_set = xbg.DMatrix(X_test, y_test, enable_categorical=True)

params = {'objective': 'binary:logistic', 'tree_method': 'hist'}
n=1000
evals = [(train_set, "train"), (test_set, "validation")]

model = xbg.train (
    params, 
    train_set,
    num_boost_round = n,
    evals = evals,
    verbose_eval = 50,
    early_stopping_rounds = 50
)

model.save_model("xgb_model.bin")

