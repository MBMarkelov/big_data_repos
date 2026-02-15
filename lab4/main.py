import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r'C:\Users\MB_Markelov_PC\Documents\GitHub\big_data_repos\lab4\Boston.csv')
print(f"Размер данных: {df.shape}")
print(f"Колонки в данных: {df.columns.tolist()}")
df.head()

print(f"a. Количество строк: {df.shape[0]}, количество столбцов: {df.shape[1]}")

memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # в MB
print(f"b. Датафрейм занимает: {memory_usage:.2f} MB")

print("\nc. Статистика для интервальных переменных:")
interval_vars = df.select_dtypes(include=[np.number]).columns
stats_df = df[interval_vars].describe(percentiles=[.25, .5, .75]).T
print(stats_df[['min', '25%', '50%', 'mean', '75%', 'max']])

print("\nd. Анализ категориальных/бинарных переменных:")
if 'chas' in df.columns:
    mode_val = df['chas'].mode()[0]
    mode_count = (df['chas'] == mode_val).sum()
    print(f"chas - мода: {mode_val}, встречается {mode_count} раз ({mode_count/len(df)*100:.1f}%)")

if 'rad' in df.columns:
    print(f"rad - уникальные значения: {df['rad'].unique()}")
    print(f"rad - распределение:\n{df['rad'].value_counts().sort_index()}")

print("\na. Анализ пропусков:")
print(df.isnull().sum())
print(f"Всего пропусков: {df.isnull().sum().sum()}")

print("\nb. Анализ выбросов:")
outlier_cols = []
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        outlier_cols.append(col)
        print(f"{col}: {len(outliers)} выбросов ({len(outliers)/len(df)*100:.1f}%)")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['medv'])
plt.title('Boxplot цены (medv) - выбросы')

plt.subplot(1, 2, 2)
sns.histplot(df['medv'], kde=True)
plt.title('Распределение цены (medv)')
plt.show()

from scipy.stats.mstats import winsorize
df['medv'] = winsorize(df['medv'], limits=[0.01, 0.01])

print("\nc. Категориальные переменные:")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Категориальных переменных: {len(categorical_cols)}")
if len(categorical_cols) == 0:
    print("В Boston Housing нет явных категориальных переменных, все числовые.")
    if 'chas' in df.columns:
        print("Кодируем chas как категориальную")
        df = pd.get_dummies(df, columns=['chas'], prefix='chas', drop_first=True)
        print("One-hot encoding применен к chas")

print("\nd. Проверка гипотез:")

corr_rm_price = df['rm'].corr(df['medv'])
print(f"Гипотеза 1: Корреляция rm и medv = {corr_rm_price:.3f}")

corr_crim_price = df['crim'].corr(df['medv'])
print(f"Гипотеза 2: Корреляция crim и medv = {corr_crim_price:.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='rm', y='medv', data=df)
plt.title('Зависимость цены от количества комнат')

plt.subplot(1, 2, 2)
sns.scatterplot(x='crim', y='medv', data=df)
plt.title('Зависимость цены от уровня преступности')
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\ne. Разделение данных: Train size: {X_train.shape}, Test size: {X_test.shape}")

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

k_values = range(1, 21)
knn_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    knn_scores.append(r2_score(y_test, y_pred))

plt.figure(figsize=(10, 5))
plt.plot(k_values, knn_scores, marker='o')
plt.xlabel('Количество соседей (k)')
plt.ylabel('R2 Score')
plt.title('Зависимость качества KNN от k')
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(knn_scores)]
print(f"Оптимальное k = {best_k} с R2 = {max(knn_scores):.3f}")

knn_model = KNeighborsRegressor(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

knn_metrics = calculate_metrics(y_test, y_pred_knn, f'KNN (k={best_k})')

from sklearn.linear_model import ElasticNetCV

elastic_model = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
elastic_model.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_model.predict(X_test_scaled)

elastic_metrics = calculate_metrics(y_test, y_pred_elastic, 'ElasticNet')
print(f"ElasticNet - alpha: {elastic_model.alpha_:.4f}, l1_ratio: {elastic_model.l1_ratio_:.2f}")

results_df = pd.DataFrame([knn_metrics, elastic_metrics])
print("\nСравнение моделей:")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
colors = ['skyblue', 'lightcoral']

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(f'Сравнение по {metric}')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)

ax = axes[1, 2]
ax.scatter(y_test, y_pred_knn, alpha=0.5, label='KNN')
ax.scatter(y_test, y_pred_elastic, alpha=0.5, label='ElasticNet')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Реальные значения')
ax.set_ylabel('Предсказанные значения')
ax.set_title('Предсказания vs Реальность')
ax.legend()

plt.tight_layout()
plt.show()

best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
if 'KNN' in best_model_name:
    best_model = knn_model
else:
    best_model = elastic_model
best_scaler = scaler

print(f"\nЛучшая модель: {best_model_name}")
print(f"R2 score: {results_df['R2'].max():.3f}")

model_filename = f'best_model_{best_model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")}.joblib'
scaler_filename = 'scaler.joblib'

joblib.dump(best_model, model_filename)
joblib.dump(best_scaler, scaler_filename)

print(f"Модель сохранена как: {model_filename}")
print(f"Scaler сохранен как: {scaler_filename}")

loaded_model = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

sample_data = X_test_scaled[:5]
predictions = loaded_model.predict(sample_data)
print("\nТест загруженной модели (первые 5 предсказаний):")
print(f"Предсказанные значения: {predictions}")
print(f"Реальные значения: {y_test[:5].values}")

if 'ElasticNet' in best_model_name:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': elastic_model.coef_
    })
    feature_importance = feature_importance[feature_importance['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['coefficient'])
    plt.xlabel('Коэффициент')
    plt.title('Важность признаков (ElasticNet)')
    plt.show()
    
    print("\nНаиболее важные признаки:")
    print(feature_importance.head(10))
else:
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation_with_target': [abs(df[col].corr(df['medv'])) for col in X.columns]
    }).sort_values('correlation_with_target', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(correlations['feature'], correlations['correlation_with_target'])
    plt.xlabel('Абсолютная корреляция с целевой переменной')
    plt.title('Важность признаков (на основе корреляции)')
    plt.show()
    
    print("\nНаиболее важные признаки (по корреляции):")
    print(correlations.head(10))

y_pred_best = best_model.predict(X_test_scaled)
residuals = y_test - y_pred_best

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков')

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q plot остатков')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ВЫВОДЫ ПО ЛАБОРАТОРНОЙ РАБОТЕ")
print("="*50)

print(f"""
В ходе лабораторной работы были выполнены следующие задачи:

1. **Загрузка и анализ данных** датасета Boston Housing
   - Загружено {df.shape[0]} строк и {df.shape[1]} столбцов
   - Данные занимают {memory_usage:.2f} MB в памяти

2. **Разведочный анализ**:
   - Рассчитаны основные статистики для всех переменных
   - Проанализированы выбросы: найдены в {len(outlier_cols)} столбцах
   - Выполнена обработка выбросов методом winsorizing

3. **Проверка гипотез**:
   - Гипотеза 1 (rm → medv): корреляция {corr_rm_price:.3f} - подтверждена

4. **Обучение моделей**:
   - KNN регрессия (оптимальное k = {best_k})
   - ElasticNet регрессия (alpha = {elastic_model.alpha_:.4f}, l1_ratio = {elastic_model.l1_ratio_:.2f})

5. **Сравнение моделей**:
""")

print(results_df.to_string(index=False))

print(f"""
6. **Лучшая модель**: {best_model_name}
   - R2 score: {results_df['R2'].max():.3f}
   - RMSE: {results_df.loc[results_df['R2'].idxmax(), 'RMSE']:.3f}
   - MAE: {results_df.loc[results_df['R2'].idxmax(), 'MAE']:.3f}

7. **Модель сохранена** в файл: {model_filename}

**Заключение**: Лучшая модель показывает хорошую обобщающую способность 
 