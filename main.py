
#importing libraries  
import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
import lightgbm as lgb 
  
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = {
    'rooms': [3, 4, 2, 5, 6],
    'area': [120, 150, 80, 200, 180],
    'year_built': [1990, 2000, 1980, 2015, 2010],
    'price': [200000, 250000, 150000, 300000, 280000]
}

df = pd.DataFrame(data) 
  
# Display the first few rows of the DataFrame to provide a data preview. 
print(df.head()) 

# Podział na cechy i etykiety
X = df.drop('price', axis=1)
y = df['price']

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)
print(X_test)
print(y_test)


# Inicjalizacja modelu
model = lgb.LGBMRegressor()

# Trening modelu
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

print(y_pred)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")