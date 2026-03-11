import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Recrearea setului de date
data = {
    'sqft': [1500, 2000, 1100, 2500, 1400, 2300, 1800, 1200, 2100, 1600],
    'bedrooms': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
    'location': ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price': [300000, 400000, 200000, 500000, 280000, 450000, 360000, 250000, 420000, 320000]
}
df = pd.DataFrame(data)

# 2. Separarea caracteristicilor (X) și a țintei (y)
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']

# Codificarea variabilelor categorice
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# Împărțirea în train și test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 3. Crearea și antrenarea modelului DecisionTreeRegressor
# Setăm max_depth=3 pentru a preveni creșterea necontrolată a arborelui (supraadaptarea) pe acest set mic
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)

# 4. Evaluarea modelului pe setul de test
y_pred = tree_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=== Rezultate Evaluare (Set Test) ===")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

# 5. Interpretarea și Vizualizarea diviziunilor
plt.figure(figsize=(12, 8))
plot_tree(tree_reg,
          feature_names=X_encoded.columns.tolist(),
          filled=True,
          rounded=True,
          precision=0)
plt.title("Decision Tree Regressor - Predicția Prețului Locuințelor")
plt.savefig("decision_tree.png", bbox_inches='tight')