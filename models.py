import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd
import joblib 


df = pd.read_csv('heart.csv')
X = df.drop(columns=['target'])
y = df['target']


X = pd.get_dummies(X, drop_first=True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train,y_train)

joblib.dump(model, 'Hheart_disease_model.pkl')
print("done")
joblib.dump(scaler, 'scaler.pkl')