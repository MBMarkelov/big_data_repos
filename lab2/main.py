"""
Задача 2: Работа с PySpark
Данные: IPL (Indian Premier League)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import sqlite3
import os
from pathlib import Path

class IPLSparkAnalyzer:
    """Класс для анализа данных IPL с использованием PySpark"""
    
    def __init__(self, sqlite_path, app_name="IPL_Analysis"):
        """
        Инициализация Spark сессии
        
        Параметры:
        sqlite_path: путь к SQLite файлу
        app_name: имя приложения Spark
        """
        self.sqlite_path = sqlite_path
        self.spark = None
        self.dataframes = {}
        
        # Создание Spark сессии
        self._create_spark_session(app_name)
        
    def _create_spark_session(self, app_name):
        """Создание и настройка Spark сессии"""
        try:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.shuffle.partitions", "8") \
                .getOrCreate()
            
            print(f"✅ Spark сессия создана: {self.spark.version}")
            print(f"   Spark UI доступен по адресу: {self.spark.sparkContext.uiWebUrl}")
            
        except Exception as e:
            print(f"❌ Ошибка создания Spark сессии: {e}")
            raise
    
    def load_data_from_sqlite(self):
        """Загрузка данных из SQLite в Spark DataFrame"""
        print("\n" + "="*60)
        print("ЗАГРУЗКА ДАННЫХ ИЗ SQLite В PySpark")
        print("="*60)
        
        # Подключение к SQLite для получения списка таблиц
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"Найдено таблиц: {tables}")
        
        # Загрузка каждой таблицы
        for table in tables:
            try:
                # Чтение через pandas (промежуточный этап для Spark)
                pdf = pd.read_sql_query(f"SELECT * FROM {table}", sqlite3.connect(self.sqlite_path))
                
                # Конвертация в Spark DataFrame
                df = self.spark.createDataFrame(pdf)
                
                # Сохранение в словарь
                self.dataframes[table.lower()] = df
                
                print(f"  ✅ Загружена таблица {table}: {df.count()} строк, {len(df.columns)} колонок")
                
                # Показ схемы
                print(f"     Схема: {[f.name for f in df.schema.fields]}")
                
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {table}: {e}")
        
        print(f"\n✅ Загружено таблиц: {len(self.dataframes)}")
    
    def register_temp_views(self):
        """Регистрация временных представлений для SQL запросов"""
        for name, df in self.dataframes.items():
            df.createOrReplaceTempView(name)
        print("✅ Временные представления зарегистрированы для SQL запросов")
    
    def execute_spark_sql(self, query, query_name=""):
        """Выполнение SQL запроса в Spark"""
        print(f"\n{'='*60}")
        print(f"ЗАПРОС {query_name}")
        print(f"{'='*60}")
        print(f"SQL:\n{query}\n")
        
        try:
            result_df = self.spark.sql(query)
            
            print(f"📊 Результат: {result_df.count()} строк")
            print(f"   Колонки: {result_df.columns}")
            
            # Показ первых строк
            result_df.show(10, truncate=False)
            
            return result_df
        except Exception as e:
            print(f"❌ Ошибка выполнения запроса: {e}")
            return None
    
    def task1_join_two_tables_spark(self):
        """Запрос 1: JOIN двух таблиц с агрегацией (Spark SQL)"""
        query = """
        SELECT 
            m.season_id,
            COUNT(*) as matches_played,
            AVG(m.win_margin) as avg_win_margin,
            MAX(m.win_margin) as max_win_margin,
            MIN(m.win_margin) as min_win_margin
        FROM match m
        JOIN season s ON m.season_id = s.season_id
        GROUP BY m.season_id, s.season_year
        ORDER BY m.season_id DESC
        """
        return self.execute_spark_sql(query, "1: JOIN двух таблиц (Match + Season) с агрегацией")
    
    def task2_join_three_tables_spark(self):
        """Запрос 2: JOIN трех таблиц с агрегацией (Spark SQL)"""
        query = """
        SELECT 
            v.venue_name,
            COUNT(DISTINCT m.match_id) as total_matches,
            COUNT(DISTINCT m.season_id) as seasons_hosted,
            AVG(m.win_margin) as avg_win_margin,
            MAX(m.win_margin) as biggest_win
        FROM match m
        JOIN venue v ON m.venue_id = v.venue_id
        JOIN season s ON m.season_id = s.season_id
        GROUP BY v.venue_id, v.venue_name
        HAVING COUNT(DISTINCT m.match_id) > 10
        ORDER BY total_matches DESC
        LIMIT 10
        """
        return self.execute_spark_sql(query, "2: JOIN трех таблиц (Match + Venue + Season)")
    
    def task3_dataframe_api_analysis(self):
        """Запрос 3: Анализ с использованием DataFrame API"""
        print(f"\n{'='*60}")
        print("ЗАПРОС 3: Анализ с использованием DataFrame API")
        print(f"{'='*60}")
        
        try:
            # Получение DataFrame'ов
            matches_df = self.dataframes.get('match')
            teams_df = self.dataframes.get('team')
            
            if matches_df is None or teams_df is None:
                print("❌ Необходимые таблицы не загружены")
                return None
            
            # Анализ эффективности команд
            result_df = matches_df \
                .groupBy("team_1") \
                .agg(
                    count("*").alias("total_matches"),
                    sum(when(col("win_type") == 1, 1).otherwise(0)).alias("wins_as_team1"),
                    avg("win_margin").alias("avg_win_margin")
                ) \
                .join(teams_df, matches_df["team_1"] == teams_df["team_id"], "left") \
                .select(
                    col("team_name"),
                    col("total_matches"),
                    col("wins_as_team1"),
                    (col("wins_as_team1") / col("total_matches") * 100).alias("win_percentage"),
                    col("avg_win_margin")
                ) \
                .orderBy(col("win_percentage").desc()) \
                .limit(10)
            
            print("📊 Топ-10 команд по проценту побед (играя первой):")
            result_df.show(10, truncate=False)
            
            return result_df
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None
    
    def task4_advanced_spark_analysis(self):
        """Запрос 4: Продвинутый анализ с оконными функциями"""
        print(f"\n{'='*60}")
        print("ЗАПРОС 4: Продвинутый анализ с оконными функциями")
        print(f"{'='*60}")
        
        try:
            matches_df = self.dataframes.get('match')
            venues_df = self.dataframes.get('venue')
            
            if matches_df is None or venues_df is None:
                print("❌ Необходимые таблицы не загружены")
                return None
            
            # Анализ стадионов с ранжированием
            stadium_stats = matches_df \
                .join(venues_df, matches_df["venue_id"] == venues_df["venue_id"], "left") \
                .groupBy("venue_name") \
                .agg(
                    count("*").alias("total_matches"),
                    avg("win_margin").alias("avg_margin"),
                    max("win_margin").alias("max_margin"),
                    stddev("win_margin").alias("margin_stddev")
                ) \
                .withColumn("rank_by_matches", row_number().over(Window.orderBy(col("total_matches").desc()))) \
                .withColumn("rank_by_margin", row_number().over(Window.orderBy(col("avg_margin").desc()))) \
                .filter(col("total_matches") > 5) \
                .orderBy(col("total_matches").desc())
            
            print("📊 Статистика по стадионам с ранжированием:")
            stadium_stats.show(15, truncate=False)
            
            return stadium_stats
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None
    
    def validate_data_types(self):
        """Проверка корректности типов данных"""
        print("\n" + "="*60)
        print("ПРОВЕРКА ТИПОВ ДАННЫХ")
        print("="*60)
        
        for name, df in self.dataframes.items():
            print(f"\n📋 Таблица: {name}")
            print(f"   Количество записей: {df.count()}")
            print("   Схема данных:")
            df.printSchema()
            
            # Проверка на null значения
            print("   Null значения по колонкам:")
            for field in df.schema.fields:
                null_count = df.filter(col(field.name).isNull()).count()
                if null_count > 0:
                    print(f"     ⚠️ {field.name}: {null_count} null значений")
                else:
                    print(f"     ✅ {field.name}: нет null значений")
    
    def save_results(self, output_dir="spark_output"):
        """Сохранение результатов анализа"""
        print(f"\n{'='*60}")
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.dataframes.items():
            try:
                # Сохранение в Parquet (эффективный формат Spark)
                output_path = f"{output_dir}/{name}_data.parquet"
                df.write.mode("overwrite").parquet(output_path)
                print(f"  ✅ Сохранена таблица {name} в {output_path}")
                
                # Сохранение в CSV для совместимости
                csv_path = f"{output_dir}/{name}_data.csv"
                df.write.mode("overwrite").option("header", "true").csv(csv_path)
                print(f"  ✅ Сохранена таблица {name} в {csv_path}")
                
            except Exception as e:
                print(f"  ❌ Ошибка сохранения {name}: {e}")
    
    def get_spark_info(self):
        """Получение информации о Spark сессии"""
        print("\n" + "="*60)
        print("ИНФОРМАЦИЯ О SPARK СЕССИИ")
        print("="*60)
        
        print(f"  Версия Spark: {self.spark.version}")
        print(f"  Spark Master: {self.spark.sparkContext.master}")
        print(f"  Приложение: {self.spark.sparkContext.appName}")
        print(f"  Spark UI: {self.spark.sparkContext.uiWebUrl}")
        print(f"  Параллельность: {self.spark.sparkContext.defaultParallelism}")
    
    def stop_spark(self):
        """Остановка Spark сессии"""
        if self.spark:
            self.spark.stop()
            print("\n✅ Spark сессия остановлена")


def main_task2():
    """Главная функция для Задачи 2"""
    print("\n" + "="*60)
    print("ЗАДАЧА 2: РАБОТА С PySpark ФРЕЙМВОРКОМ")
    print("Данные: Indian Premier League (IPL)")
    print("="*60)
    
    # Путь к SQLite базе
    sqlite_path = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\database.sqlite"
    
    # Поиск файла если не найден
    if not os.path.exists(sqlite_path):
        sqlite_files = list(Path(".").glob("*.sqlite"))
        if sqlite_files:
            sqlite_path = str(sqlite_files[0])
            print(f"Найден файл SQLite: {sqlite_path}")
        else:
            print(f"❌ Файл SQLite не найден: {sqlite_path}")
            return
    
    # Создание анализатора
    analyzer = IPLSparkAnalyzer(sqlite_path)
    
    try:
        # Получение информации о Spark
        analyzer.get_spark_info()
        
        # Загрузка данных
        analyzer.load_data_from_sqlite()
        
        # Проверка типов данных
        analyzer.validate_data_types()
        
        # Регистрация представлений для SQL
        analyzer.register_temp_views()
        
        # Выполнение запросов
        print("\n" + "="*60)
        print("ВЫПОЛНЕНИЕ ЗАПРОСОВ")
        print("="*60)
        
        # Запрос 1: JOIN двух таблиц
        analyzer.task1_join_two_tables_spark()
        
        # Запрос 2: JOIN трех таблиц
        analyzer.task2_join_three_tables_spark()
        
        # Запрос 3: DataFrame API анализ
        analyzer.task3_dataframe_api_analysis()
        
        # Запрос 4: Продвинутый анализ с оконными функциями
        analyzer.task4_advanced_spark_analysis()
        
        # Сохранение результатов
        analyzer.save_results()
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.stop_spark()
    
    print("\n" + "="*60)
    print("✅ ЗАДАЧА 2 УСПЕШНО ВЫПОЛНЕНА")
    print("="*60)


# Альтернативная версия: загрузка напрямую из CSV (если нет SQLite)
class IPLSparkFromCSV:
    """Загрузка данных из CSV файлов (альтернативный вариант)"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.spark = SparkSession.builder.appName("IPL_CSV_Analysis").getOrCreate()
        self.dataframes = {}
    
    def load_csv_files(self):
        """Загрузка CSV файлов"""
        import glob
        
        csv_files = glob.glob(f"{self.data_dir}/*.csv")
        
        for csv_file in csv_files:
            name = os.path.basename(csv_file).replace('.csv', '').lower()
            