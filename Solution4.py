import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Crearea setului de date sintetic
data = {
    'sqft': [1500, 2000, 1100, 2500, 1400, 2300, 1800, 1200, 2100, 1600],
    'bedrooms': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price': [300000, 400000, 200000, 500000, 280000, 450000, 360000, 250000, 420000, 320000]
}
df = pd.DataFrame(data)

# Preprocesare
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Este crucial să standardizăm (scalăm) datele înainte de regularizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Crearea și antrenarea modelelor Ridge și Lasso
# Folosim o valoare mare a lui alpha pentru a sublinia efectul regularizării (scăderea coeficienților)
ridge_model = Ridge(alpha=10.0)
ridge_model.fit(X_train_scaled, y_train)

# Pentru Lasso, setăm un alpha mare pentru a arăta cum unii coeficienți devin exact 0
lasso_model = Lasso(alpha=10000.0)
lasso_model.fit(X_train_scaled, y_train)

# 3. Predicții și evaluare
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)

print("=== Comparația Coeficienților ===")
print(f"Caracteristici: {X_encoded.columns.tolist()}")
print(f"Coeficienți Ridge: {np.round(ridge_model.coef_, 2)}")
print(f"Coeficienți Lasso: {np.round(lasso_model.coef_, 2)}")

print("\n=== Evaluarea pe Setul de Test ===")
print(f"Ridge -> R²: {r2_score(y_test, y_pred_ridge):.4f}, MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"Lasso -> R²: {r2_score(y_test, y_pred_lasso):.4f}, MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}")