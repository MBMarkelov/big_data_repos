# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from scipy.stats import ttest_ind
import pickle

DATA_PATH = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\KR_1\advertising.csv"
df = pd.read_csv(DATA_PATH)

TARGET = 'Clicked on Ad'
print("=== Первые 5 строк ===")
print(df.head())

eda_numeric = df.select_dtypes(include=[np.number])
eda_results = pd.DataFrame(index=eda_numeric.columns)

eda_results['missing_ratio'] = df.isnull().mean()
eda_results['min'] = eda_numeric.min()
eda_results['max'] = eda_numeric.max()
eda_results['mean'] = eda_numeric.mean()
eda_results['median'] = eda_numeric.median()
eda_results['variance'] = eda_numeric.var()
eda_results['quantile_0.1'] = eda_numeric.quantile(0.1)
eda_results['quantile_0.9'] = eda_numeric.quantile(0.9)
eda_results['quartile_1'] = eda_numeric.quantile(0.25)
eda_results['quartile_3'] = eda_numeric.quantile(0.75)

print("\n=== EDA числовых признаков ===")
print(eda_results)

plt.figure(figsize=(6,4))
sns.countplot(x=df[TARGET])
plt.title("Распределение целевой переменной")
plt.show()

categorical_cols = ['City', 'Country']
if categorical_cols:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

df = df.drop(columns=['Timestamp', 'Ad Topic Line'])

# Гипотеза 1: Средний возраст пользователей, кликнувших на рекламу, отличается от тех, кто не кликнул
age_click = df[df[TARGET]==1]['Age']
age_no_click = df[df[TARGET]==0]['Age']
t_stat, p_val_age = ttest_ind(age_click, age_no_click, equal_var=False)
print(f"\nГипотеза 1: различие среднего возраста. p-value = {p_val_age:.4f}")

# Гипотеза 2: Среднее время на сайте у кликнувших отличается от не кликнувших
time_click = df[df[TARGET]==1]['Daily Time Spent on Site']
time_no_click = df[df[TARGET]==0]['Daily Time Spent on Site']
t_stat, p_val_time = ttest_ind(time_click, time_no_click, equal_var=False)
print(f"Гипотеза 2: различие времени на сайте. p-value = {p_val_time:.4f}")

# Гистограмма времени на сайте по целевому признаку
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Daily Time Spent on Site', hue=TARGET, kde=True, palette='Set1')
plt.title("Распределение времени на сайте по кликам")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Daily Internet Usage', hue=TARGET, data=df, palette='Set2')
plt.title("Age vs Daily Internet Usage (по кликам)")
plt.show()

numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[TARGET])
corr_target = numeric_df.corrwith(df[TARGET]).sort_values(ascending=False)
print("\nТоп-10 корреляций с целевой переменной:")
print(corr_target.head(10))

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nРазмеры обучающей выборки:", X_train.shape)
print("Размеры тестовой выборки:", X_test.shape)


best_k = 5
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n=== Метрики качества ===")
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"Train Accuracy: {accuracy_train:.3f}")
print(f"Test Accuracy: {accuracy_test:.3f}")
print(f"Train F1-score: {f1_train:.3f}")
print(f"Test F1-score: {f1_test:.3f}")
print(f"Test Precision: {precision_test:.3f}")
print(f"Test Recall: {recall_test:.3f}")
print(f"Test ROC-AUC: {roc_auc_test:.3f}")
print("Confusion Matrix (Test):\n", cm_test)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nМодель сохранена в knn_model.pkl")
