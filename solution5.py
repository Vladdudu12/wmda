import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Crearea și preprocesarea setului de date
data = {
    'sqft': [1500, 2000, 1100, 2500, 1400, 2300, 1800, 1200, 2100, 1600],
    'bedrooms': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price': [300000, 400000, 200000, 500000, 280000, 450000, 360000, 250000, 420000, 320000]
}
df = pd.DataFrame(data)

X = df[['sqft', 'bedrooms', 'location']]
y = df['price']
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardizarea datelor (foarte importantă pentru modelul Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Configurarea spațiului de hiperparametri
ridge = Ridge()
# Testăm mai multe ordine de mărime pentru parametrul de regularizare 'alpha'
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}

# 3. Configurarea GridSearchCV
# Folosim 'neg_mean_squared_error' deoarece GridSearch caută să maximizeze scorul.
# Minimizarea erorii este echivalentă cu maximizarea negativului erorii.
# cv=3 împarte datele de antrenament în 3 fold-uri (fiind un set mic)
grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3
)

# Rularea căutării exhaustive
grid_search.fit(X_train_scaled, y_train)

# 4. Extragerea și evaluarea celui mai bun model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred_best = best_model.predict(X_test_scaled)
r2_best = r2_score(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

print("=== Rezultatele Optimizării GridSearchCV ===")
print(f"Cei mai buni parametri găsiți: {best_params}")
print(f"Cel mai bun scor CV (Neg MSE): {grid_search.best_score_:.2f}")

print("\n=== Performanța Celui Mai Bun Model pe Setul de Test ===")
print(f"R²: {r2_best:.4f}")
print(f"MSE: {mse_best:.2f}")
print(f"MAE: {mae_best:.2f}")

print("\n=== Coeficienții Celui Mai Bun Model ===")
print(f"[{X_encoded.columns[0]}]: {best_model.coef_[0]:.2f}")
print(f"[{X_encoded.columns[1]}]: {best_model.coef_[1]:.2f}")
print(f"[{X_encoded.columns[2]}]: {best_model.coef_[2]:.2f}")