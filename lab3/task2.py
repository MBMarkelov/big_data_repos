import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve)

print("="*60)
print("ЗАДАЧА 2: АНАЛИЗ ДАННЫХ INDIAN PREMIER LEAGUE (IPL)")
print("="*60)

db_path = r"C:\Users\MB_Markelov_PC\Documents\GitHub\big_data_repos\lab1\data\database.sqlite"

conn = sqlite3.connect(db_path)
print(f"\nУспешное подключение к базе данных: {db_path}")

match_df = pd.read_sql_query("SELECT * FROM Match", conn)
team_df = pd.read_sql_query("SELECT * FROM Team", conn)
season_df = pd.read_sql_query("SELECT * FROM Season", conn)
venue_df = pd.read_sql_query("SELECT * FROM Venue", conn)
win_by_df = pd.read_sql_query("SELECT * FROM Win_By", conn)
toss_df = pd.read_sql_query("SELECT * FROM Toss_Decision", conn)

print(f"\nЗагружена таблица Match: {match_df.shape[0]} строк, {match_df.shape[1]} колонок")
print(f"Загружена таблица Team: {team_df.shape[0]} строк")
print(f"Загружена таблица Season: {season_df.shape[0]} строк")

print("\n[1] ФОРМИРОВАНИЕ ДАТАСЕТА ДЛЯ КЛАССИФИКАЦИИ")

team_dict = dict(zip(team_df['Team_Id'], team_df['Team_Name']))
match_df['Team1_Name'] = match_df['Team_1'].map(team_dict)
match_df['Team2_Name'] = match_df['Team_2'].map(team_dict)
match_df['Match_Winner_Name'] = match_df['Match_Winner'].map(team_dict)
season_dict = dict(zip(season_df['Season_Id'], season_df['Season_Year']))
match_df['Season_Year'] = match_df['Season_Id'].map(season_dict)

df = match_df.copy()

df['Target_Win_Team1'] = (df['Match_Winner'] == df['Team_1']).astype(int)

df = df[df['Outcome_type'] != 3] 
df = df[df['Match_Winner'].notna()]

print(f"Размер датасета после фильтрации: {df.shape}")
print(f"Распределение целевой переменной (победа Team1):")
print(df['Target_Win_Team1'].value_counts(normalize=True))

print("\n[2] РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")

print(f"\nA. Размерность датафрейма: {df.shape[0]} строк, {df.shape[1]} колонок")

memory_usage = df.memory_usage(deep=True).sum() / 1024**2
print(f"B. Память: {memory_usage:.2f} МБ")

numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nC. Статистика для числовых переменных:")
print(df[numeric_cols].describe())

categorical_cols = df.select_dtypes(include=['object']).columns
print("\nD. Категориальные переменные и их мода:")
for col in categorical_cols[:5]:
    mode_val = df[col].mode()[0] if not df[col].mode().empty else "None"
    mode_count = (df[col] == mode_val).sum() if mode_val != "None" else 0
    print(f"  {col}: мода = '{mode_val}' (встречается {mode_count} раз)")

plt.figure(figsize=(6,4))
sns.countplot(x='Target_Win_Team1', data=df)
plt.title('Распределение целевой переменной (Победа Team 1)')
plt.xlabel('Победа Team 1 (0 = Нет, 1 = Да)')
plt.ylabel('Количество матчей')
plt.show()

print("\n[3] ПОДГОТОВКА ДАТАСЕТА К ПОСТРОЕНИЮ МОДЕЛЕЙ")

feature_columns = [
    'Season_Id',           
    'Toss_Winner',         
    'Toss_Decide',         
    'Win_Type',           
    'Win_Margin'       
]

X = df[feature_columns].copy()
y = df['Target_Win_Team1'].copy()

print(f"Исходные признаки: {feature_columns}")

print("\nA. Анализ пропусков:")
print(X.isnull().sum())
X = X.dropna()  
y = y.loc[X.index]  
print(f"После удаления пропусков: {X.shape[0]} строк")

print("\nB. Анализ выбросов:")
Q1 = X['Win_Margin'].quantile(0.25)
Q3 = X['Win_Margin'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = X[(X['Win_Margin'] < lower_bound) | (X['Win_Margin'] > upper_bound)]
print(f"Выбросы в Win_Margin: {len(outliers)} строк ({len(outliers)/len(X)*100:.1f}%)")
print("Вывод: Оставляем выбросы, т.к. большая маржа победы - важный признак")

print("\nC. Кодирование категориальных переменных:")
categorical_features = ['Toss_Winner', 'Toss_Decide', 'Win_Type']

for col in categorical_features:
    print(f"  {col}: уникальных значений = {X[col].nunique()}")
    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    X.drop(col, axis=1, inplace=True)

print(f"Размер X после кодирования: {X.shape}")

print("\nD. Разделение на train/test:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

print("\n[4] НОРМАЛИЗАЦИЯ ДАННЫХ (StandardScaler)")
print("Нормализация важна для KNN и SVM")

numeric_features = ['Season_Id', 'Win_Margin']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("Нормализация выполнена для признаков:", numeric_features)

print("\n[5] ОБУЧЕНИЕ МОДЕЛЕЙ")

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    results[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    }
    print(f"✓ Модель {name} обучена")

print("\n[6] ОЦЕНКА КАЧЕСТВА АЛГОРИТМОВ")
print("="*60)

metrics_df = pd.DataFrame()

for name, res in results.items():
    y_pred = res['y_pred']
    y_proba = res['y_proba']
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = np.nan
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy (A): {acc:.4f}")
    print(f"Precision (P): {prec:.4f}")
    print(f"Recall (R): {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'CM - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    metrics_df.loc[name, 'Accuracy'] = acc
    metrics_df.loc[name, 'Precision'] = prec
    metrics_df.loc[name, 'Recall'] = rec
    metrics_df.loc[name, 'F1-score'] = f1
    metrics_df.loc[name, 'ROC-AUC'] = roc_auc

print("\n--- Сводная таблица метрик ---")
print(metrics_df.round(4))

plt.figure(figsize=(8,6))
for name, res in results.items():
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        auc_val = metrics_df.loc[name, 'ROC-AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison - IPL Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n[7] ВЫБОР ОПТИМАЛЬНОГО АЛГОРИТМА")

best_model = metrics_df['F1-score'].idxmax()
best_f1 = metrics_df.loc[best_model, 'F1-score']

print(f"Лучшая модель по F1-score: {best_model} (F1 = {best_f1:.4f})")

if best_model == 'Logistic Regression':
    print("Logistic Regression показывает лучший результат благодаря:")
    print("   - Хорошей интерпретируемости")
    print("   - Устойчивости к небольшому количеству признаков")
    print("   - Эффективности на линейно разделимых данных")
elif best_model == 'SVM (RBF)':
    print("SVM показывает лучший результат благодаря:")
    print("   - Способности находить сложные нелинейные границы")
    print("   - Устойчивости к выбросам (благодаря регуляризации)")
else:
    print(f"{best_model} показывает конкурентный результат")

print("\n[8] УЛУЧШЕНИЕ АЛГОРИТМА (GridSearchCV)")

if best_model == 'KNN (k=5)':
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    base_model = KNeighborsClassifier()
elif best_model == 'SVM (RBF)':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    base_model = SVC(probability=True, random_state=42)
else:  
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    base_model = LogisticRegression(random_state=42, max_iter=1000)

print(f"Оптимизация модели: {best_model}")
print(f"Сетка параметров: {param_grid}")

grid_search = GridSearchCV(
    base_model, param_grid, cv=5, 
    scoring='f1_weighted', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nЛучшие параметры: {grid_search.best_params_}")
print(f"Лучший F1-score на CV: {grid_search.best_score_:.4f}")

best_model_optimized = grid_search.best_estimator_
y_pred_optimized = best_model_optimized.predict(X_test_scaled)
f1_optimized = f1_score(y_test, y_pred_optimized, average='weighted')

print(f"F1-score на тесте (оптимизированный): {f1_optimized:.4f}")
print(f"Улучшение: +{f1_optimized - metrics_df.loc[best_model, 'F1-score']:.4f}")

print("\n[9] ИДЕИ ДЛЯ УЛУЧШЕНИЯ АЛГОРИТМА")
print("-" * 50)
print("""
1. ДОБАВЛЕНИЕ НОВЫХ ПРИЗНАКОВ:
   - Форма команд (последние 5 матчей)
   - История встреч команд друг с другом
   - Статистика игроков (бестсмены, боулеры)
   - Погодные условия (если доступны)
   - Домашний стадион (преимущество своей площадки)

2. ФИЧИ ИЗ ДРУГИХ ТАБЛИЦ:
   - Ball_by_Ball: агрегированная статистика по матчам
   - Player_Match: индивидуальные достижения игроков
   - Batsman_Scored: тоталы ранов

3. УЛУЧШЕНИЕ МОДЕЛИ:
   - Ансамблевые методы (Random Forest, XGBoost)
   - Стекинг нескольких моделей
   - Балансировка классов через SMOTE

4. ИНЖЕНЕРИЯ ПРИЗНАКОВ:
   - Создание коэффициентов (например, Win_Margin / среднюю маржу)
   - Взаимодействие признаков (полиномиальные фичи)
   - Целевое кодирование для команд (mean target encoding)
""")

print("\nЛабораторная работа 3 (Задача 2 с IPL) выполнена на сложность Medium!")

conn.close()
print("Соединение с БД закрыто.")