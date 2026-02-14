"""
Лабораторная работа №2
Вариант 4: Indian Premier League SQLite Database
Сложность: Medium
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("АНАЛИЗ ДАННЫХ INDIAN PREMIER LEAGUE (IPL)")
print("=" * 80)

print("\n[1] ЗАГРУЗКА ДАННЫХ")
print("-" * 40)

# Подключение к базе данных SQLite
DB_PATH = r"C:\Users\MB_Markelov_PC\Documents\GitHub\big_data_repos\lab1\data\database.sqlite"
conn = sqlite3.connect(DB_PATH)

# Загружаем все таблицы
matches = pd.read_sql_query("SELECT * FROM Match", conn)
teams = pd.read_sql_query("SELECT * FROM Team", conn)
players = pd.read_sql_query("SELECT * FROM Player", conn)
season = pd.read_sql_query("SELECT * FROM Season", conn)
venue = pd.read_sql_query("SELECT * FROM Venue", conn)
win_by = pd.read_sql_query("SELECT * FROM Win_By", conn)
toss_decision = pd.read_sql_query("SELECT * FROM Toss_Decision", conn)

print(f"Таблица 'Match': {matches.shape[0]} строк, {matches.shape[1]} столбцов")
print(f"Колонки Match: {list(matches.columns)}")
print(f"Таблица 'Team': {teams.shape[0]} строк")
print(f"Таблица 'Player': {players.shape[0]} строк")
print(f"Таблица 'Season': {season.shape[0]} строк")
print(f"Таблица 'Venue': {venue.shape[0]} строк")

print("\n[2] ПРЕОБРАЗОВАНИЕ ДАННЫХ И ДОБАВЛЕНИЕ ВЫЧИСЛЯЕМЫХ СТОЛБЦОВ")
print("-" * 40)

df = matches.copy()

team_dict = dict(zip(teams['Team_Id'], teams['Team_Name']))
df['Team_1_Name'] = df['Team_1'].map(team_dict)
df['Team_2_Name'] = df['Team_2'].map(team_dict)
df['Match_Winner_Name'] = df['Match_Winner'].map(team_dict)
df['Toss_Winner_Name'] = df['Toss_Winner'].map(team_dict)

season_dict = dict(zip(season['Season_Id'], season['Season_Year']))
df['Season_Year'] = df['Season_Id'].map(season_dict)

win_type_dict = dict(zip(win_by['Win_Id'], win_by['Win_Type']))
df['Win_Type_Name'] = df['Win_Type'].map(win_type_dict)

toss_dict = dict(zip(toss_decision['Toss_Id'], toss_decision['Toss_Name']))
df['Toss_Decision_Name'] = df['Toss_Decide'].map(toss_dict)

df['Match_Date'] = pd.to_datetime(df['Match_Date'])
df['Year'] = df['Match_Date'].dt.year
df['Month'] = df['Match_Date'].dt.month
df['Day'] = df['Match_Date'].dt.day

df['Was_Result'] = (df['Outcome_type'] == 1).astype(int)

df['Win_Margin_Category'] = pd.cut(df['Win_Margin'], 
                                    bins=[0, 10, 30, 50, 100], 
                                    labels=['Small', 'Medium', 'Large', 'Very Large'])

df['High_Margin'] = (df['Win_Margin'] > 30).astype(int)

df['Season_Period'] = pd.cut(df['Season_Year'], 
                              bins=[2007, 2010, 2013, 2016, 2020], 
                              labels=['2008-2010', '2011-2013', '2014-2016', '2017-2020'])

print("Добавлены вычисляемые столбцы:")
print("- Team_1_Name, Team_2_Name, Match_Winner_Name: названия команд")
print("- Season_Year, Year, Month, Day: временные признаки")
print("- Win_Type_Name, Toss_Decision_Name: расшифровки")
print("- Was_Result: бинарный признак наличия результата")
print("- Win_Margin_Category: категоризация размера победы")
print("- High_Margin: бинарный признак большой победы")
print("- Season_Period: периоды сезонов")

print(f"\nИтоговый датафрейм: {df.shape[0]} строк, {df.shape[1]} столбцов")

print("\n[3] КОЛИЧЕСТВО СТРОК И СТОЛБЦОВ")
print("-" * 40)
print(f"Строк: {df.shape[0]}")
print(f"Столбцов: {df.shape[1]}")
print(f"Названия столбцов: {list(df.columns)}")

print("\n[4] РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print("-" * 40)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_cols = [col for col in numerical_cols if 'Id' not in col and 'id' not in col]

print(f"\nЧисловые переменные ({len(numerical_cols)}): {numerical_cols}")
print(f"Категориальные переменные ({len(categorical_cols)}): {categorical_cols}")

print("\n--- Анализ числовых переменных ---")
numerical_stats = []

for col in numerical_cols:
    if col in df.columns and df[col].notna().any():
        missing_pct = df[col].isna().mean() * 100
        
        stats_dict = {
            'Переменная': col,
            'Доля пропусков (%)': round(missing_pct, 2),
            'Минимум': round(df[col].min(), 2),
            'Максимум': round(df[col].max(), 2),
            'Среднее': round(df[col].mean(), 2),
            'Медиана': round(df[col].median(), 2),
            'Дисперсия': round(df[col].var(), 2),
            'Квантиль 0.1': round(df[col].quantile(0.1), 2),
            'Квантиль 0.9': round(df[col].quantile(0.9), 2),
            'Квартиль 1': round(df[col].quantile(0.25), 2),
            'Квартиль 3': round(df[col].quantile(0.75), 2)
        }
        numerical_stats.append(stats_dict)

numerical_df = pd.DataFrame(numerical_stats)
print(numerical_df.to_string(index=False))

print("\n--- Анализ категориальных переменных ---")
categorical_stats = []

for col in categorical_cols:
    if col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        
        n_unique = df[col].nunique()
        
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
        
        stats_dict = {
            'Переменная': col,
            'Доля пропусков (%)': round(missing_pct, 2),
            'Уникальных значений': n_unique,
            'Мода': str(mode_val)[:30] + '...' if len(str(mode_val)) > 30 else str(mode_val)
        }
        categorical_stats.append(stats_dict)

categorical_df = pd.DataFrame(categorical_stats)
print(categorical_df.to_string(index=False))

print("\n[5] ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ")
print("-" * 40)

df_clean = df.dropna(subset=['Win_Margin', 'Toss_Decision_Name', 'Season_Year'])

print("\nГИПОТЕЗА 1: Решение после жеребьевки влияет на размер победы")
print("H0: Нет разницы в размере победы между командами, выбравшими 'bat' и 'field'")
print("H1: Есть статистически значимая разница")

toss_bat = df_clean[df_clean['Toss_Decision_Name'] == 'bat']['Win_Margin'].dropna()
toss_field = df_clean[df_clean['Toss_Decision_Name'] == 'field']['Win_Margin'].dropna()

_, p_value_norm_bat = stats.shapiro(toss_bat.sample(min(100, len(toss_bat))))
_, p_value_norm_field = stats.shapiro(toss_field.sample(min(100, len(toss_field))))

print(f"\nПроверка на нормальность (тест Шапиро-Уилка):")
print(f"  bat: p-value = {p_value_norm_bat:.4f}")
print(f"  field: p-value = {p_value_norm_field:.4f}")

u_stat, p_value = mannwhitneyu(toss_bat, toss_field, alternative='two-sided')

print(f"\nСредний размер победы (bat): {toss_bat.mean():.2f}")
print(f"Средний размер победы (field): {toss_field.mean():.2f}")
print(f"U-статистика: {u_stat:.4f}")
print(f"p-значение: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("ВЫВОД: Отвергаем H0. Решение после жеребьевки статистически значимо влияет на размер победы.")
else:
    print("ВЫВОД: Нет оснований отвергнуть H0. Решение после жеребьевки не влияет на размер победы.")

print("\nГИПОТЕЗА 2: Тип победы зависит от того, кто выиграл жеребьевку")
print("H0: Нет связи между победителем жеребьевки и типом победы")
print("H1: Есть связь между этими переменными")

contingency_table = pd.crosstab(
    df_clean['Toss_Winner'] == df_clean['Match_Winner'],
    df_clean['Win_Type_Name']
)

print(f"\nТаблица сопряженности:")
print(contingency_table)

chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nХи-квадрат: {chi2:.4f}")
print(f"p-значение: {p_value_chi:.4f}")
print(f"Степени свободы: {dof}")

if p_value_chi < alpha:
    print("ВЫВОД: Отвергаем H0. Существует статистически значимая связь между победителем жеребьевки и типом победы.")
else:
    print("ВЫВОД: Нет оснований отвергнуть H0. Нет связи между победителем жеребьевки и типом победы.")

print("\n[6] ТАБЛИЦА КОРРЕЛЯЦИИ ПРИЗНАКОВ И ЦЕЛЕВОГО СТОЛБЦА")
print("-" * 40)

print("\nОБОСНОВАНИЕ ВЫБОРА ЦЕЛЕВОГО СТОЛБЦА:")
print("Целевой столбец: 'Win_Margin' (размер победы)")
print("Это ключевой показатель, характеризующий убедительность победы в матче.")
print("Мы хотим понять, какие факторы влияют на то, насколько крупно выигрывает команда.")

target = 'Win_Margin'
features_for_corr = [col for col in numerical_cols 
                     if col != target 
                     and col not in ['Match_Id', 'Team_1', 'Team_2', 'Season_Id', 'Venue_Id', 
                                    'Toss_Winner', 'Win_Type', 'Outcome_type', 'Man_of_the_Match']
                     and df[col].nunique() > 1]

corr_data = df[features_for_corr + [target]].dropna()

correlations = []
for feature in features_for_corr:
    if feature in corr_data.columns and corr_data[feature].nunique() > 1:
        corr_coef, p_val = pearsonr(corr_data[feature], corr_data[target])
        correlations.append({
            'Признак': feature,
            'Корреляция с Win_Margin': round(corr_coef, 4),
            'p-значение': round(p_val, 4),
            'Сила связи': 'Сильная' if abs(corr_coef) > 0.7 else 
                          'Средняя' if abs(corr_coef) > 0.3 else 'Слабая'
        })

corr_df = pd.DataFrame(correlations)
if not corr_df.empty:
    corr_df = corr_df.reindex(corr_df['Корреляция с Win_Margin'].abs().sort_values(ascending=False).index)
    print("\nКорреляция признаков с целевой переменной 'Win_Margin':")
    print(corr_df.to_string(index=False))
else:
    print("Недостаточно данных для корреляционного анализа")

print("\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
print("Средний размер победы по годам:")
yearly_margin = df.groupby('Year')['Win_Margin'].mean().round(2)
print(yearly_margin)

conn.close()
print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)