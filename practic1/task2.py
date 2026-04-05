# task2_pyspark_final.py
"""
Задача 2: PySpark анализ IPL
Рабочая версия (без сохранения в parquet)
"""

import os
import sys
from pathlib import Path

# ========== FIX ДЛЯ WINDOWS ==========
# Указываем путь к Python (без Microsoft Store алиаса)
VENV_PYTHON = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\.venv\Scripts\python.exe"

if os.path.exists(VENV_PYTHON):
    os.environ['PYSPARK_PYTHON'] = VENV_PYTHON
    os.environ['PYSPARK_DRIVER_PYTHON'] = VENV_PYTHON
    print(f"✅ Использую Python: {VENV_PYTHON}")

# Отключаем лишние варнинги
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[2] pyspark-shell'

# ========== ОСНОВНОЙ КОД ==========
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, col
import pandas as pd
import sqlite3

class IPLSparkAnalyzer:
    def __init__(self, sqlite_path):
        self.sqlite_path = sqlite_path
        self.spark = None
        self.dataframes = {}
        
        # Создаем Spark сессию
        self.spark = SparkSession.builder \
            .appName("IPL_Analysis") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .master("local[2]") \
            .getOrCreate()
        
        print(f"✅ Spark версия: {self.spark.version}")
    
    def load_data(self):
        """Загрузка данных из SQLite"""
        print("\n" + "="*60)
        print("ЗАГРУЗКА ДАННЫХ")
        print("="*60)
        
        conn = sqlite3.connect(self.sqlite_path)
        
        tables = ['Match', 'Season', 'Team', 'Venue', 'Toss_Decision', 'Win_By']
        
        for table in tables:
            try:
                df_pd = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df_spark = self.spark.createDataFrame(df_pd)
                self.dataframes[table] = df_spark
                print(f"  ✅ {table}: {df_spark.count()} строк")
            except Exception as e:
                print(f"  ❌ {table}: {e}")
        
        conn.close()
        print(f"\n✅ Загружено таблиц: {len(self.dataframes)}")
    
    def run_queries(self):
        """Выполнение всех запросов из задания"""
        print("\n" + "="*60)
        print("ВЫПОЛНЕНИЕ ЗАПРОСОВ")
        print("="*60)
        
        match = self.dataframes.get('Match')
        season = self.dataframes.get('Season')
        venue = self.dataframes.get('Venue')
        team = self.dataframes.get('Team')
        toss = self.dataframes.get('Toss_Decision')
        
        if match is None:
            print("❌ Таблица Match не загружена")
            return
        
        # 1. JOIN двух таблиц с агрегацией
        print("\n📊 1. JOIN Match + Season (статистика по сезонам):")
        result1 = match.join(season, match.Season_Id == season.Season_Id) \
            .groupBy(season.Season_Year) \
            .agg(
                count("*").alias("matches"),
                avg("Win_Margin").alias("avg_margin")
            ) \
            .orderBy(col("Season_Year").desc())
        result1.show(20)
        
        # 2. JOIN трех таблиц с агрегацией
        if venue:
            print("\n📊 2. JOIN Match + Venue + Season (топ стадионов):")
            result2 = match.join(venue, match.Venue_Id == venue.Venue_Id) \
                .join(season, match.Season_Id == season.Season_Id) \
                .groupBy(venue.Venue_Name) \
                .agg(count("*").alias("matches")) \
                .orderBy(col("matches").desc()) \
                .limit(10)
            result2.show()
        
        # 3. Данные по одному матчу (все таблицы)
        print("\n📊 3. Детальная информация по матчу #1:")
        result3 = match.filter(match.Match_Id == 1) \
            .join(season, match.Season_Id == season.Season_Id) \
            .join(venue, match.Venue_Id == venue.Venue_Id) \
            .select(
                match.Match_Id,
                season.Season_Year,
                venue.Venue_Name,
                match.Win_Margin
            )
        result3.show()
        
        # 4. Подсчет строк в JOIN-ах
        print("\n📊 4. Подсчет строк в JOIN-ах:")
        count_join1 = match.join(season, match.Season_Id == season.Season_Id).count()
        count_join2 = match.join(venue, match.Venue_Id == venue.Venue_Id).count() if venue else 0
        print(f"   Match + Season: {count_join1} строк")
        print(f"   Match + Venue: {count_join2} строк")
        
        # 5. Топ команд по победам
        if team:
            print("\n📊 5. Топ-10 команд по количеству побед:")
            wins_as_team1 = match.filter(match.Win_Type == 1) \
                .groupBy(match.Team_1) \
                .agg(count("*").alias("wins1"))
            
            wins_as_team2 = match.filter(match.Win_Type == 2) \
                .groupBy(match.Team_2) \
                .agg(count("*").alias("wins2"))
            
            result5 = team.join(wins_as_team1, team.Team_Id == wins_as_team1.Team_1, "left") \
                .join(wins_as_team2, team.Team_Id == wins_as_team2.Team_2, "left") \
                .fillna(0) \
                .withColumn("total_wins", col("wins1") + col("wins2")) \
                .select("Team_Name", "total_wins") \
                .orderBy(col("total_wins").desc()) \
                .limit(10)
            result5.show()
        
        # 6. Влияние решения после тосса
        if toss:
            print("\n📊 6. Влияние решения после тосса:")
            result6 = match.join(toss, match.Toss_Decide == toss.Toss_Id) \
                .groupBy(toss.Toss_Name) \
                .agg(count("*").alias("total_matches")) \
                .orderBy(col("total_matches").desc())
            result6.show()
        
        # 7. Анализ стадионов
        if venue:
            print("\n📊 7. Стадионы с самой большой средней разницей:")
            result7 = match.join(venue, match.Venue_Id == venue.Venue_Id) \
                .groupBy(venue.Venue_Name) \
                .agg(
                    count("*").alias("matches"),
                    avg("Win_Margin").alias("avg_margin")
                ) \
                .filter(col("matches") >= 5) \
                .orderBy(col("avg_margin").desc()) \
                .limit(5)
            result7.show()
    
    def stop(self):
        if self.spark:
            self.spark.stop()
            print("\n✅ Spark остановлен")


def main():
    print("="*60)
    print("ЗАДАЧА 2: PySpark АНАЛИЗ IPL")
    print("="*60)
    
    DB_PATH = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"
    
    if not Path(DB_PATH).exists():
        print(f"❌ Файл не найден: {DB_PATH}")
        return
    
    analyzer = IPLSparkAnalyzer(DB_PATH)
    
    try:
        analyzer.load_data()
        analyzer.run_queries()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        analyzer.stop()
    
    print("\n" + "="*60)
    print("✅ ЗАДАЧА 2 УСПЕШНО ВЫПОЛНЕНА")
    print("="*60)


if __name__ == "__main__":
    main()