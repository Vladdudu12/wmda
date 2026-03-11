import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

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

# 2. Scalarea caracteristicilor (Esențială pentru kNN)
# kNN se bazează pe distanța euclidiană dintre puncte.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Testarea unor valori diferite pentru k
k_values = [1, 3, 5, 7]

print("=== Evaluarea performanței pentru diverse valori ale lui k ===")
print(f"{'k':<4} | {'R² (Antrenament)':<17} | {'R² (Test)':<12} | {'MAE (Test)'}")
print("-" * 55)

for k in k_values:
    # Inițializarea și antrenarea modelului kNN Regressor
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Predicții
    y_pred_train = knn.predict(X_train_scaled)
    y_pred_test = knn.predict(X_test_scaled)

    # Calcularea metricilor
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"{k:<4} | {r2_train:<17.4f} | {r2_test:<12.4f} | {mae_test:.2f}")