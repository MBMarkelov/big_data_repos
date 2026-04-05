# check_db_structure.py
import sqlite3
import pandas as pd

db_path = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\database.sqlite"  # или полный путь к твоему файлу

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Получаем список всех таблиц
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("="*60)
print("ТАБЛИЦЫ В БАЗЕ ДАННЫХ:")
print("="*60)

for table in tables:
    table_name = table[0]
    
    # Получаем количество строк
    cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
    count = cursor.fetchone()[0]
    
    # Получаем структуру таблицы
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]
    
    print(f"\n📊 Таблица: '{table_name}'")
    print(f"   Строк: {count}")
    print(f"   Колонки: {col_names[:5]}")
    if len(col_names) > 5:
        print(f"   ... и еще {len(col_names) - 5} колонок")

conn.close()