import sqlite3
import pandas as pd
from pathlib import Path

# Путь к базе данных
DB_PATH = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"

def get_db_connection():
    """Получение соединения с БД"""
    return sqlite3.connect(DB_PATH)

def load_matches():
    """Загрузка данных о матчах"""
    conn = get_db_connection()
    matches = pd.read_sql_query("SELECT * FROM Match", conn)
    conn.close()
    return matches

def load_teams():
    """Загрузка данных о командах"""
    conn = get_db_connection()
    teams = pd.read_sql_query("SELECT * FROM Team", conn)
    conn.close()
    return teams

def load_seasons():
    """Загрузка данных о сезонах"""
    conn = get_db_connection()
    seasons = pd.read_sql_query("SELECT * FROM Season", conn)
    conn.close()
    return seasons

def load_venues():
    """Загрузка данных о стадионах"""
    conn = get_db_connection()
    venues = pd.read_sql_query("SELECT * FROM Venue", conn)
    conn.close()
    return venues

def get_season_stats():
    """Статистика по сезонам"""
    query = """
    SELECT 
        s.Season_Year,
        COUNT(*) as matches,
        ROUND(AVG(m.Win_Margin), 1) as avg_margin
    FROM Match m
    JOIN Season s ON m.Season_Id = s.Season_Id
    GROUP BY s.Season_Year
    ORDER BY s.Season_Year DESC
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

def get_top_venues():
    """Топ-5 стадионов по матчам"""
    query = """
    SELECT 
        v.Venue_Name,
        COUNT(*) as matches
    FROM Match m
    JOIN Venue v ON m.Venue_Id = v.Venue_Id
    GROUP BY v.Venue_Name
    ORDER BY matches DESC
    LIMIT 5
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

def get_top_teams():
    """Топ-5 команд по победам"""
    query = """
    SELECT 
        t.Team_Name,
        COUNT(CASE WHEN m.Win_Type = 1 AND m.Team_1 = t.Team_Id THEN 1
                   WHEN m.Win_Type = 2 AND m.Team_2 = t.Team_Id THEN 1 END) as wins
    FROM Match m
    JOIN Team t ON t.Team_Id IN (m.Team_1, m.Team_2)
    GROUP BY t.Team_Name
    ORDER BY wins DESC
    LIMIT 5
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

def get_toss_stats():
    """Статистика по тоссу"""
    query = """
    SELECT 
        td.Toss_Name,
        COUNT(*) as total_matches
    FROM Match m
    JOIN Toss_Decision td ON m.Toss_Decide = td.Toss_Id
    GROUP BY td.Toss_Name
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

def get_matches_with_details():
    """Получение матчей с деталями для EDA"""
    conn = get_db_connection()
    matches = pd.read_sql_query("SELECT * FROM Match", conn)
    seasons = pd.read_sql_query("SELECT * FROM Season", conn)
    venues = pd.read_sql_query("SELECT * FROM Venue", conn)
    teams = pd.read_sql_query("SELECT * FROM Team", conn)
    conn.close()
    
    matches_with_seasons = matches.merge(seasons, on='Season_Id')
    matches_with_venues = matches.merge(venues, on='Venue_Id')
    
    return matches, matches_with_seasons, matches_with_venues, teams