import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Crearea unui set de date redus (sintetic)
data = {
    'sqft': [1500, 2000, 1100, 2500, 1400, 2300, 1800, 1200, 2100, 1600],
    'bedrooms': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price': [300000, 400000, 200000, 500000, 280000, 450000, 360000, 250000, 420000, 320000]
}
df = pd.DataFrame(data)

# 2. Separarea caracteristicilor (X) de variabila țintă (y)
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']

# 3. Tratarea variabilelor categorice
# Folosim 'drop_first=True' pentru a evita capcana variabilelor dummy (coliniaritatea perfectă)
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# 4. Împărțirea datelor în set de antrenament și set de testare
# Alocăm 20% din date pentru testare
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Crearea și antrenarea modelului de regresie liniară
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Realizarea predicțiilor pe setul de test
y_pred = model.predict(X_test)

# 7. Evaluarea modelului folosind metrici de bază
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Afișarea rezultatelor
print("=== Rezultate Evaluare ===")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

print("\n=== Interpretare Model ===")
print("Coeficienți (sqft, bedrooms, location_cityB):", model.coef_)
print("Intercept:", model.intercept_)