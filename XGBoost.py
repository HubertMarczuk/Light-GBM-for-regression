# Importowanie bibliotek
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import xgboost as xgb

# Importowanie metod
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

# Wczytywanie danych
data = pd.read_csv("Concrete_Data.csv",decimal=',',sep=';')
df = pd.DataFrame(data) 
  
# Wyświetlenie kilku pierwszych wierszy ramki, aby wyświetlić podgląd danych
print(df.head()) 

# Podział na cechy i etykiety
X = df.drop('Strength', axis=1)
y = df['Strength']

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Wyświetlenie podzielonego zbioru danych
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Inicjalizacja modelu
model = xgb.XGBRegressor(verbose=-100, n_estimators=20, learning_rate=0.1, max_leaves=31)

# Trening modelu
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Wyświetlanie wartości dopasowanych przez model i wartości docelowych
i=0
for value in y_test:
    y_pred[i] = round(y_pred[i],2)
    print(value,"  ", y_pred[i])
    i+=1

# Wyliczanie metryk oceny modelu
mae = round(mean_absolute_error(y_test, y_pred),4)
mse = round(mean_squared_error(y_test, y_pred),4)
rmse = round(root_mean_squared_error(y_test, y_pred),4)
r2 = round(r2_score(y_test, y_pred),4)

# Wyświetlanie metryk oceny modelu
print(f"Średni błąd bezwzględny: {mae}")
print(f"Błąd średniokwadratowy: {mse}")
print(f"Średnia kwadratowa błędów: {rmse}")
print(f"Współczynnik determinacji R^2: {r2}")

# Wykres rzeczywistych vs przewidywanych wartości
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.title('Rzeczywiste vs Przewidywane wartości')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

# Wykres reszt (residual plot)
plt.figure()
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Przewidywane wartości')
plt.ylabel('Reszty')
plt.title('Wykres reszt')
plt.axhline(y=0, color='r', linestyle='--')

# Histogram reszt
plt.figure()
plt.hist(residuals, bins=30)
plt.xlabel('Reszty')
plt.ylabel('Częstotliwość')
plt.title('Histogram reszt')

# Analiza ważności cech (Feature Importance)
xgb.plot_importance(model, max_num_features=10)
plt.title('Ważność cech')

# Krzywa uczenia się (Learning Curve)
plt.figure()
model = xgb.XGBRegressor(n_estimators=20, learning_rate=0.1, max_leaves=31)
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 20), verbose=0)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
plt.plot(train_sizes, train_scores_mean, label='Błąd treningu')
plt.plot(train_sizes, test_scores_mean, label='Błąd walidacji')
plt.xlabel('Liczba próbek treningowych')
plt.ylabel('Błąd')
plt.title('Krzywa uczenia się')
plt.legend()

plt.show()