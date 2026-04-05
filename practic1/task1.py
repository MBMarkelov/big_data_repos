# task1_sqlite_final.py
"""
Задача 1: Работа с SQLite
Данные: IPL (Indian Premier League)
"""

import sqlite3
import pandas as pd
from pathlib import Path

class SQLiteIPLAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Подключение к SQLite"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Подключено к SQLite: {self.db_path}")
            return True
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False
    
    def execute_query(self, query, query_name=""):
        """Выполнение SQL запроса"""
        print(f"\n{'='*60}")
        print(f"ЗАПРОС {query_name}")
        print(f"{'='*60}")
        print(query)
        print()
        
        try:
            df = pd.read_sql_query(query, self.conn)
            print(f"Результат: {len(df)} строк")
            if len(df) > 0:
                print(df.to_string(index=False, max_rows=20))
            if len(df) > 20:
                print(f"... и еще {len(df) - 20} строк")
            return df
        except Exception as e:
            print(f" Ошибка: {e}")
            return None
    
    def query1_join_two_tables(self):
        """1. JOIN двух таблиц (Match + Season)"""
        query = """
        SELECT 
            s.Season_Year,
            COUNT(*) as matches_played,
            ROUND(AVG(m.Win_Margin), 1) as avg_win_margin,
            MAX(m.Win_Margin) as max_win_margin,
            MIN(m.Win_Margin) as min_win_margin
        FROM Match m
        JOIN Season s ON m.Season_Id = s.Season_Id
        GROUP BY s.Season_Year
        ORDER BY s.Season_Year DESC
        """
        return self.execute_query(query, "1: JOIN Match + Season")
    
    def query2_join_three_tables(self):
        """2. JOIN трех таблиц (Match + Venue + Season)"""
        query = """
        SELECT 
            v.Venue_Name,
            COUNT(DISTINCT m.Match_Id) as total_matches,
            ROUND(AVG(m.Win_Margin), 1) as avg_win_margin,
            MAX(m.Win_Margin) as biggest_win
        FROM Match m
        JOIN Venue v ON m.Venue_Id = v.Venue_Id
        JOIN Season s ON m.Season_Id = s.Season_Id
        GROUP BY v.Venue_Name
        ORDER BY total_matches DESC
        LIMIT 10
        """
        return self.execute_query(query, "2: JOIN Match + Venue + Season")
    
    def query3_one_match_details(self):
        """3. Данные по одному матчу"""
        query = """
        SELECT 
            m.Match_Id,
            s.Season_Year,
            t1.Team_Name as team_1_name,
            t2.Team_Name as team_2_name,
            v.Venue_Name,
            td.Toss_Name as toss_decision,
            wb.Win_Type as win_by_type,
            m.Win_Margin
        FROM Match m
        JOIN Season s ON m.Season_Id = s.Season_Id
        JOIN Team t1 ON m.Team_1 = t1.Team_Id
        JOIN Team t2 ON m.Team_2 = t2.Team_Id
        JOIN Venue v ON m.Venue_Id = v.Venue_Id
        LEFT JOIN Toss_Decision td ON m.Toss_Decide = td.Toss_Id
        LEFT JOIN Win_By wb ON m.Win_Type = wb.Win_Id
        WHERE m.Match_Id = 1
        """
        return self.execute_query(query, "3: Детали матча #1")
    
    def query4_count_joined_rows(self):
        """4. Подсчет строк в JOIN-ах"""
        query = """
        SELECT 'Match_Season' as join_type, COUNT(*) as total_rows
        FROM Match m JOIN Season s ON m.Season_Id = s.Season_Id
        UNION ALL
        SELECT 'Match_Venue' as join_type, COUNT(*) as total_rows
        FROM Match m JOIN Venue v ON m.Venue_Id = v.Venue_Id
        UNION ALL
        SELECT 'Match_Team1' as join_type, COUNT(*) as total_rows
        FROM Match m JOIN Team t ON m.Team_1 = t.Team_Id
        """
        return self.execute_query(query, "4: Подсчет строк")
    
    def query5_team_performance(self):
        """5. Рейтинг команд по победам"""
        query = """
        SELECT 
            t.Team_Name,
            COUNT(*) as matches_played,
            SUM(CASE 
                WHEN m.Win_Type = 1 AND m.Team_1 = t.Team_Id THEN 1
                WHEN m.Win_Type = 2 AND m.Team_2 = t.Team_Id THEN 1 
                ELSE 0 
            END) as wins
        FROM Match m
        JOIN Team t ON t.Team_Id IN (m.Team_1, m.Team_2)
        GROUP BY t.Team_Name
        ORDER BY wins DESC
        LIMIT 10
        """
        return self.execute_query(query, "5: Топ-10 команд по победам")
    
    def query6_toss_impact(self):
        """6. Влияние тосса на результат"""
        query = """
        SELECT 
            td.Toss_Name as toss_decision,
            COUNT(*) as total_matches,
            SUM(CASE 
                WHEN m.Toss_Winner = m.Match_Winner THEN 1 ELSE 0 
            END) as wins_after_toss_win,
            ROUND(100.0 * SUM(CASE 
                WHEN m.Toss_Winner = m.Match_Winner THEN 1 ELSE 0 
            END) / COUNT(*), 1) as win_percentage
        FROM Match m
        JOIN Toss_Decision td ON m.Toss_Decide = td.Toss_Id
        GROUP BY td.Toss_Name
        """
        return self.execute_query(query, "6: Влияние решения после тосса")
    
    def query7_venue_analysis(self):
        """7. Анализ стадионов"""
        query = """
        SELECT 
            v.Venue_Name,
            COUNT(*) as matches_count,
            ROUND(AVG(m.Win_Margin), 1) as avg_win_margin,
            MAX(m.Win_Margin) as max_win_margin,
            MIN(m.Win_Margin) as min_win_margin
        FROM Match m
        JOIN Venue v ON m.Venue_Id = v.Venue_Id
        GROUP BY v.Venue_Name
        HAVING COUNT(*) >= 5
        ORDER BY avg_win_margin DESC
        LIMIT 5
        """
        return self.execute_query(query, "7: Топ стадионов по разнице побед")
    
    def run_all_queries(self):
        """Запуск всех запросов"""
        self.query1_join_two_tables()
        self.query2_join_three_tables()
        self.query3_one_match_details()
        self.query4_count_joined_rows()
        self.query5_team_performance()
        self.query6_toss_impact()
        self.query7_venue_analysis()
    
    def create_dump(self, output_file="sqlite_dump.sql"):
        """Создание дампа"""
        print(f"\n{'='*60}")
        print("СОЗДАНИЕ ДАМПА")
        print(f"{'='*60}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("-- SQLite Database Dump\n\n")
                
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    schema = cursor.fetchone()
                    if schema and schema[0]:
                        f.write(f"{schema[0]};\n\n")
            
            print(f"Дамп сохранен в {output_file}")
        except Exception as e:
            print(f"Ошибка: {e}")
    
    def close(self):
        if self.conn:
            self.conn.close()
            print("\nСоединение закрыто")


def main():
    print("\n" + "="*60)
    print("ЗАДАЧА 1: РАБОТА С SQLite")
    print("Данные: Indian Premier League (IPL)")
    print("="*60)
    
    db_path = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"

    
    if not db_path:
        print("Файл database.sqlite не найден")
        return
    
    analyzer = SQLiteIPLAnalyzer(db_path)
    
    try:
        if analyzer.connect():
            analyzer.run_all_queries()
            analyzer.create_dump()
    finally:
        analyzer.close()
    
    print("\n" + "="*60)
    print("ЗАДАЧА 1 ВЫПОЛНЕНА")
    print("="*60)


if __name__ == "__main__":
    main()