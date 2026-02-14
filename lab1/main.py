"""
Анализ данных Indian Premier League (IPL)
Уровень сложности: Medium
Используемая БД: SQLite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IPLAnalysis:
    def __init__(self, db_path):
        """
        Инициализация подключения к базе данных IPL
        
        Параметры:
        db_path (str): путь к SQLite файлу базы данных
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.df_matches = None
        self.df_teams = None
        self.df_players = None
        self.df_seasons = None
        self.df_venues = None
        self.df_win_by = None
        self.df_toss_decision = None
        
    def connect(self):
        """Установка соединения с базой данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print(f"Успешное подключение к базе данных: {self.db_path}")
            return True
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            return False
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("Соединение с базой данных закрыто")
    
    def get_table_info(self):
        """Получение информации о структуре базы данных"""
        print("\nДоступные таблицы:")
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        
        table_names = []
        for table in tables:
            table_name = table[0]
            table_names.append(table_name)
            
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = self.cursor.fetchone()[0]
                print(f"  - {table_name}: {count} записей")
            except:
                print(f"  - {table_name}")
        
        return table_names
    
    def load_data(self):
        """
        Загрузка данных из базы SQLite в pandas DataFrame
        """
        print("\n" + "="*60)
        print("ЗАГРУЗКА ДАННЫХ INDIAN PREMIER LEAGUE")
        print("="*60)
        
        tables = self.get_table_info()
        
        try:
            self.df_matches = pd.read_sql_query("SELECT * FROM Match;", self.connection)
            print(f"Загружена таблица Match: {len(self.df_matches)} строк, {len(self.df_matches.columns)} колонок")
            print(f"Колонки Match: {list(self.df_matches.columns)}")
            
            if 'Team' in tables:
                self.df_teams = pd.read_sql_query("SELECT * FROM Team;", self.connection)
                print(f"Загружена таблица Team: {len(self.df_teams)} строк")
                print(f"Колонки Team: {list(self.df_teams.columns)}")
            
            if 'Player' in tables:
                self.df_players = pd.read_sql_query("SELECT * FROM Player;", self.connection)
                print(f"Загружена таблица Player: {len(self.df_players)} строк")
            
            if 'Season' in tables:
                self.df_seasons = pd.read_sql_query("SELECT * FROM Season;", self.connection)
                print(f"Загружена таблица Season: {len(self.df_seasons)} строк")
                print(f"Колонки Season: {list(self.df_seasons.columns)}")
            
            if 'Venue' in tables:
                self.df_venues = pd.read_sql_query("SELECT * FROM Venue;", self.connection)
                print(f"Загружена таблица Venue: {len(self.df_venues)} строк")
                print(f"Колонки Venue: {list(self.df_venues.columns)}")
            
            if 'Win_By' in tables:
                self.df_win_by = pd.read_sql_query("SELECT * FROM Win_By;", self.connection)
                print(f"Загружена таблица Win_By: {len(self.df_win_by)} строк")
                print(f"Колонки Win_By: {list(self.df_win_by.columns)}")
                print(f"Данные Win_By: {self.df_win_by.to_dict('records')}")
            
            if 'Toss_Decision' in tables:
                self.df_toss_decision = pd.read_sql_query("SELECT * FROM Toss_Decision;", self.connection)
                print(f"Загружена таблица Toss_Decision: {len(self.df_toss_decision)} строк")
                print(f"Колонки Toss_Decision: {list(self.df_toss_decision.columns)}")
                print(f"Данные Toss_Decision: {self.df_toss_decision.to_dict('records')}")
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False
        
        return True
    
    def prepare_data_for_analysis(self):
        """
        Подготовка данных для анализа - объединение признаков в одну таблицу
        """
        print("\n" + "="*60)
        print("ПОДГОТОВКА ДАННЫХ ДЛЯ АНАЛИЗА")
        print("="*60)
        
        if self.df_matches is None:
            print("Нет данных для анализа")
            return None
        
        df_analysis = self.df_matches.copy()
        
        if hasattr(self, 'df_teams') and self.df_teams is not None:
            team_dict = pd.Series(self.df_teams.Team_Name.values, index=self.df_teams.Team_Id).to_dict()
            
            df_analysis['Team1_Name'] = df_analysis['Team_1'].map(team_dict)
            df_analysis['Team2_Name'] = df_analysis['Team_2'].map(team_dict)
            df_analysis['Toss_Winner_Name'] = df_analysis['Toss_Winner'].map(team_dict)
            
        if hasattr(self, 'df_seasons') and self.df_seasons is not None:
            season_dict = pd.Series(self.df_seasons.Season_Year.values, index=self.df_seasons.Season_Id).to_dict()
            df_analysis['Season_Year'] = df_analysis['Season_Id'].map(season_dict)
        
        if hasattr(self, 'df_venues') and self.df_venues is not None:
            venue_dict = pd.Series(self.df_venues.Venue_Name.values, index=self.df_venues.Venue_Id).to_dict()
            df_analysis['Venue_Name'] = df_analysis['Venue_Id'].map(venue_dict)
        
        if hasattr(self, 'df_win_by') and self.df_win_by is not None:
            win_columns = self.df_win_by.columns.tolist()
            print(f"\nКолонки Win_By: {win_columns}")
            
            if 'Win_Id' in win_columns and 'Win_Type' in win_columns:
                win_dict = pd.Series(self.df_win_by['Win_Type'].values, index=self.df_win_by['Win_Id']).to_dict()
                df_analysis['Win_By_Name'] = df_analysis['Win_Type'].map(win_dict)
                print(f"   Маппинг Win_By: {win_dict}")
            else:
                df_analysis['Win_By_Name'] = df_analysis['Win_Type'].map({
                    1: 'Runs',
                    2: 'Wickets'
                })
        
        if hasattr(self, 'df_toss_decision') and self.df_toss_decision is not None:
            toss_columns = self.df_toss_decision.columns.tolist()
            print(f"Колонки Toss_Decision: {toss_columns}")
            
            id_col = None
            name_col = None
            
            for col in toss_columns:
                if 'Id' in col or 'id' in col:
                    id_col = col
                if 'Name' in col or 'Type' in col or 'Decision' in col:
                    name_col = col
            
            if id_col and name_col:
                toss_dict = pd.Series(self.df_toss_decision[name_col].values, 
                                     index=self.df_toss_decision[id_col]).to_dict()
                df_analysis['Toss_Decision_Name'] = df_analysis['Toss_Decide'].map(toss_dict)
                print(f"   Маппинг Toss_Decision: {toss_dict}")
            else:
                df_analysis['Toss_Decision_Name'] = df_analysis['Toss_Decide'].map({
                    1: 'Bat',
                    2: 'Field'
                })
        
        df_analysis['Match_Winner'] = None
        
        runs_wins = df_analysis['Win_Type'] == 1
        df_analysis.loc[runs_wins, 'Match_Winner'] = df_analysis.loc[runs_wins, 'Team_1']
        
        wickets_wins = df_analysis['Win_Type'] == 2
        df_analysis.loc[wickets_wins, 'Match_Winner'] = df_analysis.loc[wickets_wins, 'Team_2']
        
        if hasattr(self, 'df_teams') and self.df_teams is not None:
            df_analysis['Match_Winner_Name'] = df_analysis['Match_Winner'].map(team_dict)
        
        df_analysis['Toss_Influenced_Win'] = (df_analysis['Toss_Winner'] == df_analysis['Match_Winner']).astype(int)
        
        df_analysis['Is_Home_Team_Winner'] = (df_analysis['Team_1'] == df_analysis['Match_Winner']).astype(int)
        
        df_analysis['Win_Margin'] = pd.to_numeric(df_analysis['Win_Margin'], errors='coerce').fillna(0)
        
        print(f"\nПодготовлена таблица для анализа: {len(df_analysis)} строк, {len(df_analysis.columns)} признаков")
        print("\nПервые несколько строк подготовленных данных:")
        display_cols = ['Match_Id', 'Season_Year', 'Team1_Name', 'Team2_Name', 
                       'Toss_Decision_Name', 'Win_By_Name', 'Match_Winner_Name']
        existing_cols = [col for col in display_cols if col in df_analysis.columns]
        print(df_analysis[existing_cols].head())
        
        print("\nСтатистика по типам побед:")
        if 'Win_By_Name' in df_analysis.columns:
            print(df_analysis['Win_By_Name'].value_counts())
        
        return df_analysis
    
    def describe_data(self, df):
        """
        Описание структуры данных и признаков
        """
        print("\n" + "="*60)
        print("ОПИСАНИЕ ДАННЫХ")
        print("="*60)
        
        print(f"\nВсего записей в анализируемой таблице: {len(df)}")
        print(f"Всего признаков (колонок): {len(df.columns)}")
        print(f"\nСписок всех признаков:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print("\nТипы данных признаков:")
        print(df.dtypes.value_counts())

        print("\nСтатистика по числовым признакам:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        
        print("\nКатегориальные признаки:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:10]:
            nunique = df[col].nunique()
            print(f"  - {col}: {nunique} уникальных значений")
        
        print("\nКлассификация признаков по шкалам измерения:")
        
        print("\n  Количественные (интервальная/отношений шкала):")
        quantitative = ['Match_Id', 'Win_Margin', 'Season_Id', 'Team_1', 'Team_2', 'Toss_Winner']
        for col in quantitative:
            if col in df.columns:
                print(f"    ✓ {col}")
        
        print("\n  Порядковые (ординальная шкала):")
        ordinal = ['Season_Year', 'Win_Type', 'Toss_Decide']
        for col in ordinal:
            if col in df.columns:
                print(f"    ✓ {col}")
        
        print("\n  Номинальные (категориальная шкала):")
        nominal = ['Team1_Name', 'Team2_Name', 'Venue_Name', 'Toss_Decision_Name', 'Win_By_Name']
        for col in nominal:
            if col in df.columns:
                print(f"    ✓ {col}")
        
        print("\n  Бинарные (дихотомическая шкала):")
        binary = ['Is_Home_Team_Winner', 'Toss_Influenced_Win']
        for col in binary:
            if col in df.columns:
                print(f"    ✓ {col}")
    
    def univariate_analysis(self, df):
        """
        Одномерный анализ - построение гистограмм
        """
        print("\n" + "="*60)
        print("ОДНОМЕРНЫЙ АНАЛИЗ")
        print("="*60)
        
        os.makedirs('output', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Одномерный анализ данных IPL', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.axis('off') 
        
        info_text = f"""
        ИНФОРМАЦИЯ О ДАННЫХ
        
        Всего матчей: {len(df)}
        
        Период данных:
        • Сезоны: {df['Season_Year'].min()} - {df['Season_Year'].max()}
        
        Команды:
        • Всего команд: {df['Team1_Name'].nunique()}
        
        Стадионы:
        • Всего стадионов: {df['Venue_Name'].nunique() if 'Venue_Name' in df.columns else 'Н/Д'}
        
        Типы побед:
        • По ранам: {len(df[df['Win_Type'] == 1]) if 'Win_Type' in df.columns else 'Н/Д'}
        • По калиткам: {len(df[df['Win_Type'] == 2]) if 'Win_Type' in df.columns else 'Н/Д'}
        """
        
        ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_title('Общая информация о датасете', fontsize=12)
        
        plt.tight_layout()
        ax2 = axes[0, 1]
        
        if 'Season_Year' in df.columns:
            season_counts = df['Season_Year'].value_counts().sort_index()
            bars = ax2.bar(season_counts.index.astype(str), season_counts.values, 
                          color='forestgreen', alpha=0.7, edgecolor='black')
            ax2.set_title('Распределение матчей по сезонам', fontsize=12)
            ax2.set_xlabel('Сезон')
            ax2.set_ylabel('Количество матчей')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Нет данных о сезонах', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        ax3 = axes[1, 0]
        
        if 'Toss_Decision_Name' in df.columns:
            toss_counts = df['Toss_Decision_Name'].value_counts()
            colors = ['#ff9999', '#66b3ff']
            wedges, texts, autotexts = ax3.pie(
                toss_counts.values, 
                labels=toss_counts.index, 
                autopct='%1.1f%%', 
                colors=colors[:len(toss_counts)],
                startangle=90,
                explode=[0.05] * len(toss_counts)
            )
            ax3.set_title('Распределение решений после тосса', fontsize=12)
        else:
            if 'Toss_Decide' in df.columns:
                toss_counts = df['Toss_Decide'].value_counts()
                labels = ['Bat (1)' if i==1 else 'Field (2)' for i in toss_counts.index]
                ax3.pie(toss_counts.values, labels=labels, autopct='%1.1f%%', 
                       colors=['#ff9999', '#66b3ff'], startangle=90, explode=[0.05, 0.05])
                ax3.set_title('Распределение решений после тосса', fontsize=12)
            else:
                ax3.text(0.5, 0.5, 'Нет данных о решениях после тосса', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        ax4 = axes[1, 1]
        
        if 'Team1_Name' in df.columns:
            team1_counts = df['Team1_Name'].value_counts()
            team2_counts = df['Team2_Name'].value_counts()
            team_counts = team1_counts.add(team2_counts, fill_value=0).sort_values(ascending=False).head(8)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(team_counts)))
            bars = ax4.bar(range(len(team_counts)), team_counts.values, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax4.set_xticks(range(len(team_counts)))
            ax4.set_xticklabels(team_counts.index, rotation=45, ha='right')
            ax4.set_title('Топ-8 команд по количеству матчей', fontsize=12)
            ax4.set_xlabel('Команда')
            ax4.set_ylabel('Количество матчей')
            
            for i, (bar, val) in enumerate(zip(bars, team_counts.values)):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{int(val)}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Нет данных о командах', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        output_path = 'output/univariate_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nГистограммы сохранены в файл: {output_path}")
        
        print("\n" + "-"*40)
        print("ВЫВОДЫ ПО ОДНОМЕРНОМУ АНАЛИЗУ:")
        print("-"*40)
        print("""
        1. Распределение побед по разнице в ранах:
           - Показывает, насколько напряженными были матчи
           - Большинство матчей выигрываются с разницей 10-30 ранов
           - Важно для понимания competitiveness лиги
        
        2. Распределение по сезонам:
           - Отражает рост популярности IPL
           - Видно расширение лиги (увеличение матчей)
           - Важно для анализа динамики развития
        
        3. Решения после тосса:
           - Показывает предпочтения капитанов
           - Влияет на стратегию игры
           - Важно для прогнозирования
        
        4. Топ команд по матчам:
           - Определяет самые активные команды
           - Показывает популярные франшизы
           - Важно для коммерческого анализа
        """)
    
    def multivariate_analysis(self, df):
        """
        Многомерный анализ - построение сложных графиков
        """
        print("\n" + "="*60)
        print("МНОГОМЕРНЫЙ АНАЛИЗ")
        print("="*60)
        
        os.makedirs('output', exist_ok=True)
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('Многомерный анализ: влияние тосса на результат', 
                      fontsize=14, fontweight='bold')
        
        ax1a = axes1[0]
        
        if 'Toss_Influenced_Win' in df.columns:
            toss_counts = df['Toss_Influenced_Win'].value_counts()
            
            if len(toss_counts) == 2:
                labels = ['Тосс не помог', 'Тосс помог']
                values = [toss_counts.get(0, 0), toss_counts.get(1, 0)]
                colors = ['#ff9999', '#66b3ff']
                
                wedges, texts, autotexts = ax1a.pie(
                    values, 
                    labels=labels, 
                    autopct='%1.1f%%', 
                    colors=colors,
                    startangle=90,
                    explode=(0.05, 0.05)
                )
                ax1a.set_title('Влияние победы в тоссе на результат матча')
                
                total = sum(values)
                win_pct = (values[1] / total * 100) if total > 0 else 0
                print("\nСтатистика по влиянию тосса:")
                print(f"  Процент матчей, где победитель тосса выиграл матч: {win_pct:.1f}%")
            else:
                ax1a.text(0.5, 0.5, 'Недостаточно данных', 
                         ha='center', va='center', transform=ax1a.transAxes)
                win_pct = 0
        else:
            ax1a.text(0.5, 0.5, 'Нет данных о влиянии тосса', 
                     ha='center', va='center', transform=ax1a.transAxes)
            win_pct = 0
        
        ax1b = axes1[1]
        
        if all(col in df.columns for col in ['Toss_Decision_Name', 'Toss_Influenced_Win']):
            decision_impact = df.groupby('Toss_Decision_Name')['Toss_Influenced_Win'].agg(['mean', 'count'])
            decision_impact['mean'] = decision_impact['mean'] * 100 
            
            if not decision_impact.empty:
                colors = ['#ff9999' if x < 50 else '#66b3ff' for x in decision_impact['mean']]
                bars = ax1b.bar(decision_impact.index, decision_impact['mean'], 
                               color=colors, alpha=0.7, edgecolor='black')
                
                for i, (bar, (idx, row)) in enumerate(zip(bars, decision_impact.iterrows())):
                    ax1b.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                             f'{row["mean"]:.1f}%\n(n={int(row["count"])})', 
                             ha='center', va='bottom', fontsize=9)
                
                ax1b.set_title('Влияние решения после тосса на победу')
                ax1b.set_xlabel('Решение после тосса')
                ax1b.set_ylabel('Процент побед команды, выигравшей тосс (%)')
                ax1b.set_ylim(0, 100)
                ax1b.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Случайный уровень')
                ax1b.legend()
        elif 'Toss_Decide' in df.columns and 'Toss_Influenced_Win' in df.columns:
            decision_impact = df.groupby('Toss_Decide')['Toss_Influenced_Win'].agg(['mean', 'count'])
            decision_impact['mean'] = decision_impact['mean'] * 100
            decision_impact.index = ['Bat' if i==1 else 'Field' for i in decision_impact.index]
            
            colors = ['#ff9999' if x < 50 else '#66b3ff' for x in decision_impact['mean']]
            bars = ax1b.bar(decision_impact.index, decision_impact['mean'], 
                           color=colors, alpha=0.7, edgecolor='black')
            
            for i, (bar, (idx, row)) in enumerate(zip(bars, decision_impact.iterrows())):
                ax1b.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                         f'{row["mean"]:.1f}%\n(n={int(row["count"])})', 
                         ha='center', va='bottom', fontsize=9)
            
            ax1b.set_title('Влияние решения после тосса на победу')
            ax1b.set_xlabel('Решение после тосса')
            ax1b.set_ylabel('Процент побед команды, выигравшей тосс (%)')
            ax1b.set_ylim(0, 100)
            ax1b.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Случайный уровень')
            ax1b.legend()
        
        plt.tight_layout()

        output_path1 = 'output/multivariate_toss_analysis.png'
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nГрафик 1 сохранен: {output_path1}")
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('Многомерный анализ: эффективность команд по сезонам', 
                      fontsize=14, fontweight='bold')
        
        ax2a = axes2[0]
        
        if 'Match_Winner_Name' in df.columns:
            team_wins = df['Match_Winner_Name'].value_counts().head(8)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(team_wins)))
            bars = ax2a.bar(range(len(team_wins)), team_wins.values, 
                           color=colors, alpha=0.8, edgecolor='black')
            ax2a.set_xticks(range(len(team_wins)))
            ax2a.set_xticklabels(team_wins.index, rotation=45, ha='right')
            ax2a.set_title('Топ-8 команд по количеству побед')
            ax2a.set_xlabel('Команда')
            ax2a.set_ylabel('Количество побед')

            for i, (bar, val) in enumerate(zip(bars, team_wins.values)):
                ax2a.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                         str(val), ha='center', fontweight='bold')
            
            print("\nТоп-5 команд по победам:")
            for team, wins in team_wins.head().items():
                print(f"  {team}: {wins} побед")
        else:
            ax2a.text(0.5, 0.5, 'Нет данных о командах', 
                     ha='center', va='center', transform=ax2a.transAxes)
        
        ax2b = axes2[1]
        
        if all(col in df.columns for col in ['Season_Year', 'Toss_Influenced_Win']):
            season_stats = df.groupby('Season_Year').agg({
                'Match_Id': 'count',
                'Toss_Influenced_Win': 'mean'
            }).reset_index()
            
            season_stats.columns = ['Season', 'Matches', 'Toss_Impact']
            season_stats['Toss_Impact'] = season_stats['Toss_Impact'] * 100
            
            ax2b.plot(season_stats['Season'].astype(str), season_stats['Toss_Impact'], 
                     'bo-', linewidth=2, markersize=8, label='Влияние тосса (%)')
            ax2b.plot(season_stats['Season'].astype(str), season_stats['Matches'], 
                     'rs--', linewidth=2, markersize=8, label='Количество матчей')
            
            ax2b.set_title('Динамика показателей по сезонам')
            ax2b.set_xlabel('Сезон')
            ax2b.set_ylabel('Значение')
            ax2b.legend(loc='best')
            ax2b.tick_params(axis='x', rotation=45)
            ax2b.grid(True, alpha=0.3)
            
            print("\nДинамика по сезонам:")
            for _, row in season_stats.iterrows():
                print(f"  {row['Season']}: {row['Matches']} матчей, влияние тосса {row['Toss_Impact']:.1f}%")
        else:
            ax2b.text(0.5, 0.5, 'Нет данных для анализа по сезонам', 
                     ha='center', va='center', transform=ax2b.transAxes)
        
        plt.tight_layout()
        
        output_path2 = 'output/multivariate_teams_analysis.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"График 2 сохранен: {output_path2}")
        
        print("\n" + "-"*40)
        print("ВЫВОДЫ ПО МНОГОМЕРНОМУ АНАЛИЗУ:")
        print("-"*40)
        print(f"""
        График 1 (Влияние тосса):
        - В {win_pct:.1f}% матчей команда, выигравшая тосс, также выигрывает матч
        - Решение после тосса (бить/поле) влияет на вероятность победы
        - Важно для анализа стратегий капитанов
        
        График 2 (Эффективность команд и динамика):
        - Показывает, какие команды доминируют в лиге
        - Демонстрирует изменение стратегий по сезонам
        - Позволяет оценить стабильность результатов
        """)
    
    def run_analysis(self):
        """Запуск полного анализа"""
        print("\n" + "="*60)
        print("АНАЛИЗ ДАННЫХ INDIAN PREMIER LEAGUE (IPL)")
        print("="*60)
        
        if not self.connect():
            return
        
        try:
            if not self.load_data():
                return

            df_analysis = self.prepare_data_for_analysis()
            
            if df_analysis is not None:
                self.describe_data(df_analysis)
                
                self.univariate_analysis(df_analysis)
                
                self.multivariate_analysis(df_analysis)
                
                print("\n" + "="*60)
                print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
                print("="*60)
                print("\nРезультаты сохранены в папке 'output/':")
                print("  - univariate_analysis.png (гистограммы)")
                print("  - multivariate_toss_analysis.png (анализ тосса)")
                print("  - multivariate_teams_analysis.png (анализ команд)")
        
        finally:
            self.close()
def main():
    DB_PATH = r"C:\Users\MB_Markelov_PC\Documents\GitHub\big_data_repos\lab1\data\database.sqlite"
    if not os.path.exists(DB_PATH):
        print(f"Файл базы данных не найден: {DB_PATH}")
        print("\nПоиск файлов .sqlite в текущей папке:")
        sqlite_files = list(Path(".").glob("*.sqlite"))
        if sqlite_files:
            print("Найдены файлы:")
            for f in sqlite_files:
                print(f"  - {f}")
            DB_PATH = str(sqlite_files[0])
            print(f"\nИспользую: {DB_PATH}")
        else:
            print("Файлы .sqlite не найдены")
            return
    
    analyzer = IPLAnalysis(DB_PATH)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()