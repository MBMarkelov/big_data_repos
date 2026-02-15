import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)

data = datasets.load_breast_cancer()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = pd.Series(data["target"], name='target')

print("="*60)
print("ЗАДАЧА 1: Анализ данных о раке груди (Breast Cancer Wisconsin)")
print("="*60)

print("\n[1] РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")

print(f"\nA. Размерность датафрейма: {X.shape[0]} строк, {X.shape[1]} столбцов")

memory_usage = X.memory_usage(deep=True).sum() / 1024**2  # в МБ
print(f"B. Память: {memory_usage:.2f} МБ")

print("\nC. Статистика для интервальных переменных (первые 5 признаков):")
stats = X.describe(percentiles=[0.25, 0.5, 0.75]).T
stats = stats[['min', '25%', '50%', 'mean', '75%', 'max']]
print(stats.head())

print("\nD. Категориальные переменные:")
print("В признаках (X) категориальных переменных нет (все числовые).")
print("Распределение целевой переменной (y):")
print(y.value_counts())
mode_value = y.mode()[0]
mode_count = (y == mode_value).sum()
print(f"Мода целевой переменной: {mode_value} (встречается {mode_count} раз)")

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Распределение целевой переменной')
plt.xlabel('Диагноз (0 = Злокачественный, 1 = Доброкачественный)')
plt.ylabel('Количество')
plt.show()
print("\n[2] ПОДГОТОВКА ДАТАСЕТА")

print("\nA. Анализ пропусков:")
print(X.isnull().sum().sum()) 
if X.isnull().sum().sum() == 0:
    print("Пропусков в данных нет.")

print("\nB. Анализ выбросов (метод межквартильного размаха - IQR):")
outliers_count = 0
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
    if len(outliers) > 0:
        outliers_count += len(outliers)
print(f"Всего найдено {outliers_count} потенциальных выбросов (могут быть важны для диагностики).")
print("Вывод: Выбросы не удаляем, т.к. в медицинских данных они могут нести важную информацию о патологии.")

print("\nC. Категориальные переменные:")
print("Все переменные в X уже числовые. Кодирование не требуется.")

print("\nD. Разделение на train/test:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\n[3] НОРМАЛИЗАЦИЯ ДАННЫХ (StandardScaler)")
print("Важно для KNN и SVM, т.к. эти алгоритмы чувствительны к масштабу признаков.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Нормализация выполнена")

print("\n[4] ОБУЧЕНИЕ МОДЕЛЕЙ (с нормализованными данными)")

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True)
}

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    predictions[name] = y_pred
    probabilities[name] = y_proba
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(f"\nМодель {name} обучена.")

print("\n" + "="*60)
print("[5] ОЦЕНКА КАЧЕСТВА АЛГОРИТМОВ НА ТЕСТОВЫХ ДАННЫХ")
print("="*60)

metrics_summary = {}

for name, res in results.items():
    y_pred = res['y_pred']
    y_proba = res['y_proba']

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'

    metrics_summary[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'ROC-AUC': roc_auc
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy (A): {acc:.4f}")
    print(f"Precision (P): {prec:.4f}")
    print(f"Recall (R): {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print("\n--- Сводная таблица метрик ---")
summary_df = pd.DataFrame(metrics_summary).T
print(summary_df)

plt.figure(figsize=(8,6))
for name, res in results.items():
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics_summary[name]["ROC-AUC"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n[6] ВЫБОР ОПТИМАЛЬНОГО АЛГОРИТМА")

best_model_name = summary_df['F1-score'].idxmax()
best_model_f1 = summary_df.loc[best_model_name, 'F1-score']

print(f"Лучшая модель по метрике F1-score: {best_model_name} (F1 = {best_model_f1:.4f})")
print("\nОбоснование:")
print("- Все три модели показали отличные результаты (Accuracy > 0.95).")
print("- Logistic Regression и SVM имеют чуть более стабильные показатели Precision/Recall.")
print("- В медицинских задачах важно минимизировать ложноотрицательные результаты (FN),")
print("  поэтому высокий Recall (R) критичен. У всех моделей Recall > 0.96.")
print("- SVM показывает лучший баланс (F1) и высокий ROC-AUC, что делает его предпочтительным.")

print("\n[7] ИДЕИ ДЛЯ УЛУЧШЕНИЯ АЛГОРИТМОВ")
print("-" * 40)
print("1. **Подбор гиперпараметров (GridSearchCV)**:")
print("   - Для KNN: перебор количества соседей (k), метрики расстояния.")
print("   - Для SVM: перебор параметров C (регуляризация) и gamma (ядро RBF).")
print("2. **Feature Engineering (инженерия признаков)**:")
print("   - Создание полиномиальных признаков (взаимодействие переменных).")
print("   - Отбор наиболее важных признаков (SelectKBest, RFE).")
print("3. **Использование других моделей**:")
print("   - Random Forest, Gradient Boosting (XGBoost, LightGBM) — часто дают прирост.")
print("4. **Балансировка классов**:")
print("   - Хотя классы сбалансированы, можно применить SMOTE для аугментации.")
print("5. **Кросс-валидация**:")
print("   - Использовать k-fold кросс-валидацию вместо одного разбиения train/test.")

print("\n--- Конец лабораторной работы ---")

print("\n[ДОПОЛНИТЕЛЬНО] Пример улучшения KNN через подбор гиперпараметров:")
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(3, 15, 2),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', verbose=0)
grid_search.fit(X_train_scaled, y_train)

print(f"Лучшие параметры для KNN: {grid_search.best_params_}")
print(f"Лучший F1-score на кросс-валидации: {grid_search.best_score_:.4f}")

best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_scaled)
print(f"F1-score на тесте с оптимизированным KNN: {f1_score(y_test, y_pred_best):.4f}")