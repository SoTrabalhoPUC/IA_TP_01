from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import pandas as pd

file = pd.read_csv('winequality-white.csv', delimiter = ';')
file = file.drop(columns = ['quality'])
X = file.iloc[:, 0:10]
y = file.iloc[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1 , test_size = 0.2)
dtr = DecisionTreeRegressor(criterion = 'friedman_mse', max_depth = 10, min_samples_leaf = 4, min_samples_split = 10)
dtr.fit(X_train, y_train) 
y_pred = dtr.predict(X_test)


# Calcular as previsões do modelo nos dados de teste
y_pred = dtr.predict(X_test)

# Calcular o MSE
mse = mean_squared_error(y_test, y_pred)

# Calcular o MAE
mae = mean_absolute_error(y_test, y_pred)

# Calcular o R²
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)
