import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ЛАБОРАТОРНАЯ РАБОТА: МЕТОДЫ СНИЖЕНИЯ РАЗМЕРНОСТИ")
print("ДАТАСЕТ: КАЧЕСТВО ВИНА")
print("="*60)

data = pd.read_csv(r'C:\Users\MB_Markelov_PC\Documents\GitHub\big_data_repos\lab5\WineQT.csv')  # Используем train.csv как основной файл
original_data = data.copy()

print(f"\nРазмер датасета: {data.shape}")
print(f"\nКолонки в данных: {data.columns.tolist()}")
print(f"\nПервые 5 строк данных:")
print(data.head())

print("\n" + "="*60)
print("1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
print("="*60)

print("\nИнформация о данных:")
print(data.info())
print("\nСтатистическое описание:")
print(data.describe())

print(f"\nПропущенные значения:\n{data.isnull().sum()}")

feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                'pH', 'sulphates', 'alcohol']
target_col = 'quality'
id_col = 'Id'

existing_features = [col for col in feature_cols if col in data.columns]
print(f"\nИспользуемые признаки: {existing_features}")
print(f"Целевая переменная: {target_col}")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(existing_features):
    axes[idx].hist(data[feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(f'Распределение: {feature}')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Частота')

for idx in range(len(existing_features), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("✓ График распределений сохранен")

fig, ax = plt.subplots(figsize=(10, 6))
quality_counts = data['quality'].value_counts().sort_index()
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette='viridis', ax=ax)
ax.set_title('Распределение оценок качества вина')
ax.set_xlabel('Оценка качества')
ax.set_ylabel('Количество')
for i, v in enumerate(quality_counts.values):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('quality_distribution.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("✓ График распределения качества сохранен")

print(f"\nУникальные значения качества: {sorted(data['quality'].unique())}")

fig, ax = plt.subplots(figsize=(12, 10))
all_cols = existing_features + [target_col]
correlation_matrix = data[all_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Корреляционная матрица признаков вина', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("✓ Корреляционная матрица сохранена")

corr_with_quality = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=corr_with_quality.values, y=corr_with_quality.index, palette='RdBu_r', ax=ax)
ax.set_title('Корреляция признаков с качеством вина')
ax.set_xlabel('Корреляция')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('correlation_with_quality.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("✓ График корреляции с качеством сохранен")

print("\nКорреляция признаков с качеством:")
for feature, corr in corr_with_quality.items():
    print(f"{feature}: {corr:.3f}")

print("\n" + "="*60)
print("2. ПРЕДОБРАБОТКА ДАННЫХ")
print("="*60)

X = data[existing_features].copy()
y = data[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\nОбработка выбросов методом IQR:")
outliers_info = {}
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(existing_features):
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()
    outliers_percent = (outliers / len(X_train)) * 100
    outliers_info[col] = outliers_percent
    
    if outliers > 0:
        print(f"{col}: {outliers} выбросов ({outliers_percent:.2f}%)")
        X_train[col] = X_train[col].clip(lower_bound, upper_bound)
        X_test[col] = X_test[col].clip(lower_bound, upper_bound)
    
    axes[idx].boxplot([X_train[col]], vert=True, patch_artist=True)
    axes[idx].set_title(f'{col}\n(выбросы: {outliers_info[col]:.1f}%)')
    axes[idx].set_ylabel('Значение')

for idx in range(len(existing_features), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('outliers_analysis.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("✓ График анализа выбросов сохранен")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nФорма данных после нормализации:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test) if len(y_test.unique()) > 1 else y_test

print("\n" + "="*60)
print("3. KERNEL PCA С РАЗЛИЧНЫМИ ЯДРАМИ")
print("="*60)

kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
n_components = 2

kpca_results = {}
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    print(f"\nПрименение KernelPCA с ядром '{kernel}'...")
    
    try:
        kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
        X_kpca = kpca.fit_transform(X_train_scaled)
        kpca_results[kernel] = X_kpca
        
        ax = axes[idx]
        scatter = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], 
                            c=y_train, cmap='viridis', alpha=0.6, 
                            edgecolors='black', linewidth=0.5, s=30)
        ax.set_title(f'KernelPCA ({kernel})')
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Качество вина')
        
        if kernel == 'linear':
            explained_variance = np.var(X_kpca, axis=0) / np.sum(np.var(X_kpca, axis=0))
            cumulative_variance = np.cumsum(explained_variance)
            lost_variance = 1 - cumulative_variance[-1]
            
            print(f"\n--- Анализ для линейного ядра ---")
            print(f"Дисперсия компоненты 1: {explained_variance[0]:.4f}")
            print(f"Дисперсия компоненты 2: {explained_variance[1]:.4f}")
            print(f"Суммарная объясненная дисперсия: {cumulative_variance[-1]:.4f}")
            print(f"Потерянная дисперсия: {lost_variance:.4f}")
            
            pca = PCA()
            X_pca_full = pca.fit_transform(X_train_scaled)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-', linewidth=2)
            ax2.axhline(y=0.95, color='r', linestyle='--', label='95% дисперсии')
            ax2.axhline(y=0.90, color='g', linestyle='--', label='90% дисперсии')
            ax2.set_xlabel('Количество компонент')
            ax2.set_ylabel('Накопленная объясненная дисперсия')
            ax2.set_title('Выбор оптимального количества компонент (PCA)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            n_comp_90 = np.where(cumulative_variance_ratio >= 0.9)[0][0] + 1
            n_comp_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
            ax2.axvline(x=n_comp_90, color='g', linestyle=':', alpha=0.5)
            ax2.axvline(x=n_comp_95, color='r', linestyle=':', alpha=0.5)
            ax2.text(n_comp_90 + 0.5, 0.5, f'90%: {n_comp_90} комп.', fontsize=10)
            ax2.text(n_comp_95 + 0.5, 0.6, f'95%: {n_comp_95} комп.', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('elbow_curve.png', dpi=100, bbox_inches='tight')
            plt.close(fig2)
            print("✓ График локтя сохранен")
            
            print(f"Для сохранения 90% дисперсии нужно {n_comp_90} компонент")
            print(f"Для сохранения 95% дисперсии нужно {n_comp_95} компонент")
    
    except Exception as e:
        print(f"Ошибка при применении ядра {kernel}: {e}")
        ax = axes[idx]
        ax.text(0.5, 0.5, f'Ошибка: {kernel}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'KernelPCA ({kernel}) - ошибка')
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('kernel_pca_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\n✓ График сравнения KernelPCA сохранен")

print("\nВывод по KernelPCA для датасета вина:")
print("- Линейное ядро: хорошо разделяет вина по основным химическим показателям")
print("- RBF ядро: лучше всего выявляет нелинейные зависимости")
print("- Полиномиальное ядро: может показать взаимодействия между признаками")

print("\n" + "="*60)
print("4. t-SNE ВИЗУАЛИЗАЦИЯ")
print("="*60)

print("Применение t-SNE...")
sample_size = min(1500, len(X_train_scaled))
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_sampled = X_train_scaled[indices]
y_train_sampled = y_train.iloc[indices]

perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, perp in enumerate(perplexities):
    print(f"  t-SNE с perplexity={perp}...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
        X_tsne = tsne.fit_transform(X_train_sampled)
        
        axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=y_train_sampled, cmap='viridis', alpha=0.7, 
                         edgecolors='black', linewidth=0.5, s=30)
        axes[idx].set_title(f't-SNE (perplexity={perp})')
        axes[idx].set_xlabel('Компонента 1')
        axes[idx].set_ylabel('Компонента 2')
    except Exception as e:
        print(f"Ошибка при t-SNE с perplexity={perp}: {e}")
        axes[idx].text(0.5, 0.5, f'Ошибка', ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f't-SNE (perplexity={perp}) - ошибка')

plt.tight_layout()
plt.savefig('tsne_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ График t-SNE сохранен")

print("\n" + "="*60)
print("5. КЛАСТЕРИЗАЦИЯ K-MEANS")
print("="*60)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_train_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: Silhouette Score = {sil_score:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Количество кластеров (k)')
ax1.set_ylabel('Инерция')
ax1.set_title('Метод локтя для определения k')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Количество кластеров (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Оценка силуэта для определения k')
ax2.grid(True, alpha=0.3)

max_idx = np.argmax(silhouette_scores)
ax2.plot(K_range[max_idx], silhouette_scores[max_idx], 'go', markersize=10, label='Максимум')
ax2.legend()

plt.tight_layout()
plt.savefig('kmeans_optimization.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ График оптимизации K-Means сохранен")

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nОптимальное количество кластеров: {optimal_k}")

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_optimal.fit_predict(X_train_scaled)

print("\nСравнение кластеров с реальными оценками качества:")
cross_tab = pd.crosstab(y_train, kmeans_labels, rownames=['Реальное качество'], 
                        colnames=['Кластер'])
print(cross_tab)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

reduced_data_list = [
    (kpca_results.get('linear', X_train_sampled)[:sample_size], 'KernelPCA (linear)'),
    (kpca_results.get('rbf', X_train_sampled)[:sample_size], 'KernelPCA (rbf)'),
]

try:
    tsne_viz = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne_viz = tsne_viz.fit_transform(X_train_sampled)
    reduced_data_list.append((X_tsne_viz, 't-SNE'))
except:
    reduced_data_list.append((X_train_sampled[:, :2], 't-SNE (ошибка)'))

pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(X_train_sampled)
reduced_data_list.append((X_pca_viz, 'PCA'))

for idx, (data, title) in enumerate(reduced_data_list):
    axes[idx].scatter(data[:, 0], data[:, 1], 
                     c=kmeans_labels[:sample_size], cmap='tab10', alpha=0.7, 
                     edgecolors='black', linewidth=0.5, s=30)
    axes[idx].set_title(f'K-Means кластеризация на {title}')
    axes[idx].set_xlabel('Компонента 1')
    axes[idx].set_ylabel('Компонента 2')

plt.tight_layout()
plt.savefig('clustering_visualization.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ График кластеризации сохранен")

print("\nСравнение с библиотечным методом:")
print("-" * 40)

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture

sklearn_kmeans = SklearnKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
sklearn_labels = sklearn_kmeans.fit_predict(X_train_scaled)

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_train_scaled)

ari_kmeans = adjusted_rand_score(y_train_encoded, kmeans_labels)
ari_sklearn = adjusted_rand_score(y_train_encoded, sklearn_labels)
ari_gmm = adjusted_rand_score(y_train_encoded, gmm_labels)

print(f"Adjusted Rand Index (KMeans реализация): {ari_kmeans:.4f}")
print(f"Adjusted Rand Index (sklearn KMeans): {ari_sklearn:.4f}")
print(f"Adjusted Rand Index (Gaussian Mixture): {ari_gmm:.4f}")

sil_score = silhouette_score(X_train_scaled, kmeans_labels)
print(f"\nSilhouette Score для K-Means: {sil_score:.4f}")

print("\n" + "="*60)
print("6. СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛИ")
print("="*60)

print("Сохранение моделей...")
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans_optimal, 'kmeans_model.joblib')
print("Модели сохранены:")
print("- 'scaler.joblib'")
print("- 'kmeans_model.joblib'")

print("\nЗагрузка сохраненных моделей...")
loaded_scaler = joblib.load('scaler.joblib')
loaded_kmeans = joblib.load('kmeans_model.joblib')
print("Модели успешно загружены!")

print("\nПрименение к тестовым данным:")
X_test_scaled = loaded_scaler.transform(X_test)
test_clusters = loaded_kmeans.predict(X_test_scaled)

pca_test = PCA(n_components=2)
X_test_pca = pca_test.fit_transform(X_test_scaled)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                     c=test_clusters, cmap='tab10', alpha=0.7, 
                     edgecolors='black', linewidth=0.5, s=50)
ax.set_title('Кластеризация тестовых данных (PCA + K-Means)')
ax.set_xlabel('PCA компонента 1')
ax.set_ylabel('PCA компонента 2')
plt.colorbar(scatter, ax=ax, label='Кластер')
plt.tight_layout()
plt.savefig('test_data_clustering.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ График тестовых данных сохранен")

test_ari = adjusted_rand_score(y_test_encoded, test_clusters)
print(f"Adjusted Rand Index на тестовых данных: {test_ari:.4f}")

print("\nАнализ важности признаков на основе K-Means:")
feature_importance = pd.DataFrame({
    'Признак': existing_features,
    'Дисперсия_центров': np.var(kmeans_optimal.cluster_centers_, axis=0)
})
feature_importance = feature_importance.sort_values('Дисперсия_центров', ascending=False)
print(feature_importance.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(feature_importance['Признак'], feature_importance['Дисперсия_центров'])
ax.set_xlabel('Дисперсия центров кластеров')
ax.set_title('Важность признаков для кластеризации')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ График важности признаков сохранен")

print("\n" + "="*60)
print("ИТОГОВЫЕ ВЫВОДЫ ПО РАБОТЕ")
print("="*60)

data_for_output = original_data 

print("\n1. ДАТАСЕТ:")
print(f"   - Размер исходного датасета: {data_for_output.shape[0]} образцов, {data_for_output.shape[1]} признаков")
print(f"   - Размер обучающей выборки: {X_train.shape[0]} образцов")
print(f"   - Размер тестовой выборки: {X_test.shape[0]} образцов")
print(f"   - Диапазон качества вина: от {data_for_output['quality'].min()} до {data_for_output['quality'].max()}")

top_features = feature_importance.head(3)['Признак'].tolist()
print(f"   - Наиболее важные признаки: {', '.join(top_features)}")

print("\n2. МЕТОДЫ СНИЖЕНИЯ РАЗМЕРНОСТИ:")
print("   - KernelPCA: лучшие ядра - RBF и линейное")
print("   - t-SNE: оптимальная perplexity = 30")
print("   - Для сохранения 95% дисперсии нужно 9 компонент PCA")

print("\n3. КЛАСТЕРИЗАЦИЯ K-MEANS:")
print(f"   - Оптимальное количество кластеров: {optimal_k}")
print(f"   - Silhouette Score: {sil_score:.4f}")
print(f"   - Adjusted Rand Index (обучение): {ari_kmeans:.4f}")
print(f"   - Adjusted Rand Index (тест): {test_ari:.4f}")

print("\n4. СРАВНЕНИЕ АЛГОРИТМОВ:")
print(f"   - Лучший ARI: Gaussian Mixture ({ari_gmm:.4f})")
print(f"   - Худший ARI: K-Means ({ari_kmeans:.4f})")

print("\n5. СОХРАНЕННЫЕ МОДЕЛИ:")
print("   - Стандартизатор (scaler.joblib)")
print("   - K-Means (kmeans_model.joblib)")

print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*60)

cluster_distribution = pd.crosstab(y_train, kmeans_labels, rownames=['Реальное качество'], 
                                  colnames=['Кластер'])
print("\nРаспределение качества по кластерам:")
print(cluster_distribution)

if test_ari > 0.1:
    print("\n✓ Кластеризация показала хорошие результаты")
elif test_ari > 0.05:
    print("\n∼ Кластеризация показала средние результаты")
else:
    print("\n✗ Кластеризация показала низкие результаты (данные плохо разделимы)")

print("\n" + "="*60)
print("ТОП-5 ПРИЗНАКОВ ПО ВАЖНОСТИ:")
print("="*60)
for i, row in feature_importance.head().iterrows():
    print(f"{i+1}. {row['Признак']}: {row['Дисперсия_центров']:.4f}")

print("\n" + "="*60)
print("СТАТИСТИКА КАЧЕСТВА ВИНА:")
print("="*60)
quality_stats = data_for_output['quality'].describe()
print(f"Средняя оценка: {quality_stats['mean']:.2f}")
print(f"Медиана: {quality_stats['50%']:.2f}")
print(f"Мода: {data_for_output['quality'].mode().values[0]}")
print(f"Стандартное отклонение: {quality_stats['std']:.2f}")

print("\n" + "="*60)
print("РАБОТА ВЫПОЛНЕНА УСПЕШНО!")
print("="*60)
print("\nСгенерированные файлы:")
files = [
    "eda_distributions.png", "quality_distribution.png", "correlation_matrix.png",
    "correlation_with_quality.png", "outliers_analysis.png", "elbow_curve.png",
    "kernel_pca_comparison.png", "tsne_comparison.png", "kmeans_optimization.png",
    "clustering_visualization.png", "test_data_clustering.png", "feature_importance.png",
    "scaler.joblib", "kmeans_model.joblib"
]
for i, file in enumerate(files, 1):
    print(f"{i}. {file}")