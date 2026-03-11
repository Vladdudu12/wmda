import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Recrearea aceluiași set de date
data = {
    'sqft': [1500, 2000, 1100, 2500, 1400, 2300, 1800, 1200, 2100, 1600],
    'bedrooms': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price': [300000, 400000, 200000, 500000, 280000, 450000, 360000, 250000, 420000, 320000]
}
df = pd.DataFrame(data)

X = df[['sqft', 'bedrooms', 'location']]
y = df['price']

# 2. Tratarea variabilelor categorice (One-Hot Encoding)
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# 3. Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- A. Modelul Liniar de Bază ---
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# --- B. Modelul Liniar cu Caracteristici Polinomiale (Inginerie de caracteristici) ---
# Folosim PolynomialFeatures cu gradul 2 pentru a adăuga termeni pătratici (ex: sqft^2)
# și termeni de interacțiune (ex: sqft * bedrooms)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test) # Transformăm setul de test folosind aceleași reguli

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# De asemenea, calculăm predicțiile pe setul de antrenament pentru a verifica supraadaptarea (overfitting)
y_train_pred_poly = poly_model.predict(X_train_poly)

# --- C. Comparație și Evaluare ---
print("=== Model Liniar (De Bază) ===")
print(f"R² Test: {r2_score(y_test, y_pred_lin):.4f}")
print(f"MSE Test: {mean_squared_error(y_test, y_pred_lin):.2f}")
print(f"MAE Test: {mean_absolute_error(y_test, y_pred_lin):.2f}")

print("\n=== Model Polinomial (Gradul 2) ===")
print(f"R² Antrenament (Train): {r2_score(y_train, y_train_pred_poly):.4f}")
print(f"R² Test: {r2_score(y_test, y_pred_poly):.4f}")
print(f"MSE Test: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"MAE Test: {mean_absolute_error(y_test, y_pred_poly):.2f}")