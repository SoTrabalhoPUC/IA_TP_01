import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

# Carregar os dados
data = pd.read_csv("/content/sample_data/employee_data.csv")

# Remover colunas desnecessárias
columns_to_drop = ['EmpID', 'FirstName', 'LastName', 'ADEmail', 'DOB', 'Supervisor', 'PayZone', 'TerminationDescription']
data = data.drop(columns_to_drop, axis=1)

# Codificar os dados categóricos
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Dividir os dados em features e target
X = data.drop('Current Employee Rating', axis=1)
y = data['Current Employee Rating']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Escalonar os dados
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Balanceamento de dados usando SMOTE
print('Antes do Balanceamento: ', Counter(y_train))
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
print('Depois do balanceamento: ', Counter(y_train_sm))

# Balanceamento de dados usando SMOTETomek
print('Antes do Balanceamento: ', Counter(y_train))
smtmek = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smtmek.fit_resample(X_train, y_train)
print('Depois do balanceamento: ', Counter(y_train_smt))

# Treinamento e predição com Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes_pred = naive_bayes.predict(X_test)

naive_bayes_sm = GaussianNB()
naive_bayes_sm.fit(X_train_sm, y_train_sm)
naive_bayes_pred_sm = naive_bayes_sm.predict(X_test)

naive_bayes_smt = GaussianNB()
naive_bayes_smt.fit(X_train_smt, y_train_smt)
naive_bayes_pred_smt = naive_bayes_smt.predict(X_test)

# Treinamento e predição com Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_pred = decision_tree.predict(X_test)

decision_tree_sm = DecisionTreeClassifier(random_state=42)
decision_tree_sm.fit(X_train_sm, y_train_sm)
decision_tree_pred_sm = decision_tree_sm.predict(X_test)

decision_tree_smt = DecisionTreeClassifier(random_state=42)
decision_tree_smt.fit(X_train_smt, y_train_smt)
decision_tree_pred_smt = decision_tree_smt.predict(X_test)

# Treinamento e predição com XGBoost
xgboost = XGBClassifier(n_estimators=650, max_depth=10, learning_rate=0.01, subsample=1, random_state=0)
xgboost.fit(X_train, y_train)
xgboost_predict = xgboost.predict(X_test)

xgboost_sm = XGBClassifier(n_estimators=650, max_depth=10, learning_rate=0.01, subsample=1, random_state=0)
xgboost_sm.fit(X_train_sm, y_train_sm)
xgboost_predict_sm = xgboost_sm.predict(X_test)

xgboost_smt = XGBClassifier(n_estimators=650, max_depth=10, learning_rate=0.01, subsample=1, random_state=0)
xgboost_smt.fit(X_train_smt, y_train_smt)
xgboost_predict_smt = xgboost_smt.predict(X_test)

# Treinamento e predição com Random Forest
random = RandomForestClassifier(n_estimators=10, max_features=3, criterion='gini', random_state=0)
random.fit(X_train, y_train)
random_predict = random.predict(X_test)

random_sm = RandomForestClassifier(n_estimators=10, max_features=3, criterion='gini', random_state=0)
random_sm.fit(X_train_sm, y_train_sm)
random_predict_sm = random_sm.predict(X_test)

random_smt = RandomForestClassifier(n_estimators=10, max_features=3, criterion='gini', random_state=0)
random_smt.fit(X_train_smt, y_train_smt)
random_predict_smt = random_smt.predict(X_test)

# Relatório de classificação para cada modelo
print("Naive Bayes:")
print("Sem Balanceamento:")
print(classification_report(y_test, naive_bayes_pred))
print("Com SMOTE:")
print(classification_report(y_test, naive_bayes_pred_sm))
print("Com SMOTETomek:")
print(classification_report(y_test, naive_bayes_pred_smt))

print("\nDecision Tree:")
print("Sem Balanceamento:")
print(classification_report(y_test, decision_tree_pred))
print("Com SMOTE:")
print(classification_report(y_test, decision_tree_pred_sm))
print("Com SMOTETomek:")
print(classification_report(y_test, decision_tree_pred_smt))

print("\nXGBoost:")
print("Sem Balanceamento:")
print(classification_report(y_test, xgboost_predict))
print("Com SMOTE:")
print(classification_report(y_test, xgboost_predict_sm))
print("Com SMOTETomek:")
print(classification_report(y_test, xgboost_predict_smt))

print("\nRandom Forest:")
print("Sem Balanceamento:")
print(classification_report(y_test, random_predict))
print("Com SMOTE:")
print(classification_report(y_test, random_predict_sm))
print("Com SMOTETomek:")
print(classification_report(y_test, random_predict_smt))
