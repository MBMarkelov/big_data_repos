from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

app = FastAPI(title="IPL Analytics API")

# Путь к базе данных
DB_PATH = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"

# Глобальные переменные
matches = None
teams = None
seasons = None
venues = None
model = None
label_encoders = {}
model_trained = False

# Загрузка данных при старте
@app.on_event("startup")
async def startup_event():
    global matches, teams, seasons, venues
    print("Загрузка данных...")
    conn = sqlite3.connect(DB_PATH)
    matches = pd.read_sql_query("SELECT * FROM Match", conn)
    teams = pd.read_sql_query("SELECT * FROM Team", conn)
    seasons = pd.read_sql_query("SELECT * FROM Season", conn)
    venues = pd.read_sql_query("SELECT * FROM Venue", conn)
    conn.close()
    print(f"✅ Загружено {len(matches)} матчей, {len(teams)} команд")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPL Cricket Analytics</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
            border-radius: 15px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .nav-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }
        .nav-btn:hover, .nav-btn.active {
            background: #e94560;
            transform: translateY(-2px);
        }
        .page { display: none; }
        .page.active { display: block; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h2 {
            color: #e94560;
            margin-bottom: 20px;
            border-bottom: 2px solid #e94560;
            padding-bottom: 10px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-card h3 { font-size: 14px; margin-bottom: 10px; opacity: 0.9; }
        .metric-card p { font-size: 28px; font-weight: bold; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        th {
            background: rgba(233,69,96,0.3);
            color: #e94560;
        }
        tr:hover { background: rgba(255,255,255,0.05); }
        .chart-container { margin-top: 20px; text-align: center; }
        .chart-container img { max-width: 100%; border-radius: 10px; }
        select, button {
            padding: 10px 15px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            margin: 5px;
        }
        select { background: rgba(255,255,255,0.9); }
        button {
            background: #e94560;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            background: #ff6b8a;
            transform: translateY(-2px);
        }
        .prediction-result {
            margin-top: 20px;
            padding: 20px;
            background: rgba(233,69,96,0.2);
            border-radius: 10px;
            text-align: center;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            border-radius: 10px;
            z-index: 1000;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid #e94560;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .form-group { display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏏 IPL Cricket Analytics Dashboard</h1>
            <p>Анализ данных Indian Premier League</p>
        </div>
        
        <div class="nav">
            <button class="nav-btn active" onclick="showPage('overview')">📋 Обзор</button>
            <button class="nav-btn" onclick="showPage('eda')">📊 EDA Анализ</button>
            <button class="nav-btn" onclick="showPage('ml')">🤖 ML Модель</button>
            <button class="nav-btn" onclick="showPage('queries')">📈 SQL Запросы</button>
        </div>
        
        <div id="overview" class="page active">
            <div class="card">
                <h2>📊 Ключевые метрики</h2>
                <div class="metrics" id="metrics"></div>
            </div>
            <div class="card">
                <h2>📈 Статистика по сезонам</h2>
                <div id="season-stats"></div>
            </div>
        </div>
        
        <div id="eda" class="page">
            <div class="card">
                <h2>📅 Матчи по сезонам</h2>
                <div class="chart-container" id="season-chart"></div>
            </div>
            <div class="card">
                <h2>🏆 Распределение разницы побед</h2>
                <div class="chart-container" id="win-margin-chart"></div>
            </div>
            <div class="card">
                <h2>🏟️ Топ-10 стадионов</h2>
                <div class="chart-container" id="top-venues-chart"></div>
            </div>
            <div class="card">
                <h2>📊 Топ-5 команд по победам</h2>
                <div id="top-teams"></div>
            </div>
        </div>
        
        <div id="ml" class="page">
            <div class="card">
                <h2>🤖 Машинное обучение</h2>
                <div id="model-status"></div>
                <button onclick="trainModel()" style="margin-top: 20px;">🚀 Обучить модель</button>
            </div>
            <div class="card" id="prediction-card" style="display:none;">
                <h2>🔮 Предсказание результата матча</h2>
                <div id="prediction-controls"></div>
                <div id="prediction-result"></div>
            </div>
        </div>
        
        <div id="queries" class="page">
            <div class="card">
                <h2>📊 Статистика по сезонам</h2>
                <div id="sql-season-stats"></div>
            </div>
            <div class="card">
                <h2>🏟️ Топ-5 стадионов</h2>
                <div id="sql-top-venues"></div>
            </div>
            <div class="card">
                <h2>🏆 Топ-5 команд по победам</h2>
                <div id="sql-top-teams"></div>
            </div>
            <div class="card">
                <h2>🎲 Влияние тосса</h2>
                <div id="sql-toss-stats"></div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Загрузка...</p>
        </div>
    </div>
    
    <script>
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function showLoading() { document.getElementById('loading').style.display = 'block'; }
        function hideLoading() { document.getElementById('loading').style.display = 'none'; }
        
        async function loadStats() {
            const response = await fetch('/api/stats');
            const data = await response.json();
            document.getElementById('metrics').innerHTML = `
                <div class="metric-card"><h3>Всего матчей</h3><p>${data.matches}</p></div>
                <div class="metric-card"><h3>Команд</h3><p>${data.teams}</p></div>
                <div class="metric-card"><h3>Сезонов</h3><p>${data.seasons}</p></div>
                <div class="metric-card"><h3>Средний разрыв</h3><p>${data.avg_win_margin.toFixed(1)}</p></div>
            `;
        }
        
        async function loadSeasonStats() {
            const response = await fetch('/api/season-stats');
            const data = await response.json();
            let html = '<table><thead><tr><th>Сезон</th><th>Матчи</th><th>Средний разрыв</th></tr></thead><tbody>';
            data.forEach(row => {
                html += `<tr><td>${row.Season_Year}</td><td>${row.matches}</td><td>${row.avg_margin}</td></tr>`;
            });
            html += '</tbody></table>';
            document.getElementById('season-stats').innerHTML = html;
            document.getElementById('sql-season-stats').innerHTML = html;
        }
        
        async function loadCharts() {
            const seasonResp = await fetch('/api/season-chart');
            const seasonData = await seasonResp.json();
            document.getElementById('season-chart').innerHTML = `<img src="data:image/png;base64,${seasonData.image}">`;
            
            const marginResp = await fetch('/api/win-margin-chart');
            const marginData = await marginResp.json();
            document.getElementById('win-margin-chart').innerHTML = `<img src="data:image/png;base64,${marginData.image}">`;
            
            const venuesResp = await fetch('/api/top-venues-chart');
            const venuesData = await venuesResp.json();
            document.getElementById('top-venues-chart').innerHTML = `<img src="data:image/png;base64,${venuesData.image}">`;
        }
        
        async function loadTopTeams() {
            const response = await fetch('/api/top-teams');
            const data = await response.json();
            let html = '<table><thead><tr><th>Команда</th><th>Победы</th></tr></thead><tbody>';
            data.forEach(row => {
                html += `<tr><td>${row.Team_Name}</td><td>${row.wins}</td></tr>`;
            });
            html += '</tbody></table>';
            document.getElementById('top-teams').innerHTML = html;
            document.getElementById('sql-top-teams').innerHTML = html;
        }
        
        async function loadTopVenues() {
            const response = await fetch('/api/top-venues');
            const data = await response.json();
            let html = '<table><thead><tr><th>Стадион</th><th>Матчи</th></tr></thead><tbody>';
            data.forEach(row => {
                html += `<tr><td>${row.Venue_Name}</td><td>${row.matches}</td></tr>`;
            });
            html += '</tbody></table>';
            document.getElementById('sql-top-venues').innerHTML = html;
        }
        
        async function loadTossStats() {
            const response = await fetch('/api/toss-stats');
            const data = await response.json();
            let html = '<table><thead><tr><th>Решение</th><th>Матчи</th></tr></thead><tbody>';
            data.forEach(row => {
                html += `<tr><td>${row.Toss_Name}</td><td>${row.total_matches}</td></tr>`;
            });
            html += '</tbody></table>';
            document.getElementById('sql-toss-stats').innerHTML = html;
        }
        
        async function trainModel() {
            showLoading();
            const response = await fetch('/api/train-model', { method: 'POST' });
            const data = await response.json();
            hideLoading();
            
            if (data.success) {
                document.getElementById('model-status').innerHTML = `
                    <div style="background:rgba(0,255,0,0.2); padding:15px; border-radius:8px;">
                        ✅ ${data.message}<br>
                        📊 Обучающая выборка: ${data.train_size} матчей<br>
                        📊 Тестовая выборка: ${data.test_size} матчей
                    </div>
                `;
                document.getElementById('prediction-card').style.display = 'block';
                await loadModelInfo();
            } else {
                document.getElementById('model-status').innerHTML = `
                    <div style="background:rgba(255,0,0,0.2); padding:15px; border-radius:8px;">
                        ❌ Ошибка: ${data.error}
                    </div>
                `;
            }
        }
        
        async function loadModelInfo() {
            const response = await fetch('/api/model-info');
            const data = await response.json();
            
            if (data.trained) {
                let html = '<div>';
                html += '<select id="team1" style="width:200px"><option value="">Команда 1</option>';
                data.teams.forEach(team => html += `<option value="${team}">${team}</option>`);
                html += '</select>';
                
                html += '<select id="team2" style="width:200px"><option value="">Команда 2</option>';
                data.teams.forEach(team => html += `<option value="${team}">${team}</option>`);
                html += '</select>';
                
                html += '<select id="venue" style="width:250px"><option value="">Стадион</option>';
                data.venues.forEach(venue => html += `<option value="${venue}">${venue}</option>`);
                html += '</select>';
                
                html += '<select id="season" style="width:100px"><option value="">Сезон</option>';
                data.seasons.forEach(season => html += `<option value="${season}">${season}</option>`);
                html += '</select>';
                
                html += '<button onclick="predictMatch()">Предсказать</button>';
                html += '</div>';
                
                document.getElementById('prediction-controls').innerHTML = html;
            }
        }
        
        async function predictMatch() {
            const team1 = document.getElementById('team1').value;
            const team2 = document.getElementById('team2').value;
            const venue = document.getElementById('venue').value;
            const season = document.getElementById('season').value;
            
            if (!team1 || !team2 || !venue || !season) {
                alert('Заполните все поля');
                return;
            }
            
            if (team1 === team2) {
                alert('Команды должны быть разными');
                return;
            }
            
            showLoading();
            
            const formData = new FormData();
            formData.append('team1', team1);
            formData.append('team2', team2);
            formData.append('venue', venue);
            formData.append('season', season);
            
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await response.json();
            hideLoading();
            
            if (data.success) {
                document.getElementById('prediction-result').innerHTML = `
                    <div class="prediction-result">
                        <h3>🏆 Результат предсказания</h3>
                        <p style="font-size:24px">Победит: <strong>${data.winner}</strong></p>
                        <p>Вероятность: ${(data.probability * 100).toFixed(1)}%</p>
                    </div>
                `;
            } else {
                document.getElementById('prediction-result').innerHTML = `
                    <div class="prediction-result">❌ Ошибка: ${data.error}</div>
                `;
            }
        }
        
        loadStats();
        loadSeasonStats();
        loadCharts();
        loadTopTeams();
        loadTopVenues();
        loadTossStats();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

# API endpoints
@app.get("/api/stats")
async def get_stats():
    return {
        "matches": len(matches),
        "teams": len(teams),
        "seasons": len(seasons),
        "venues": len(venues),
        "avg_win_margin": float(matches['Win_Margin'].mean())
    }

@app.get("/api/season-stats")
async def season_stats():
    conn = sqlite3.connect(DB_PATH)
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
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

@app.get("/api/top-venues")
async def top_venues():
    conn = sqlite3.connect(DB_PATH)
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
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

@app.get("/api/top-teams")
async def top_teams():
    conn = sqlite3.connect(DB_PATH)
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
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

@app.get("/api/toss-stats")
async def toss_stats():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        td.Toss_Name,
        COUNT(*) as total_matches
    FROM Match m
    JOIN Toss_Decision td ON m.Toss_Decide = td.Toss_Id
    GROUP BY td.Toss_Name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict('records')

@app.get("/api/season-chart")
async def season_chart():
    conn = sqlite3.connect(DB_PATH)
    matches_with_seasons = pd.read_sql_query("""
        SELECT m.*, s.Season_Year 
        FROM Match m 
        JOIN Season s ON m.Season_Id = s.Season_Id
    """, conn)
    conn.close()
    
    season_counts = matches_with_seasons.groupby('Season_Year').size()
    
    plt.figure(figsize=(10, 5))
    plt.bar(season_counts.index.astype(str), season_counts.values, color='steelblue')
    plt.xlabel('Сезон')
    plt.ylabel('Количество матчей')
    plt.title('Матчи IPL по годам')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {"image": image_base64}

@app.get("/api/win-margin-chart")
async def win_margin_chart():
    conn = sqlite3.connect(DB_PATH)
    matches = pd.read_sql_query("SELECT Win_Margin FROM Match", conn)
    conn.close()
    
    plt.figure(figsize=(10, 5))
    plt.hist(matches['Win_Margin'].dropna(), bins=30, color='forestgreen', edgecolor='black')
    plt.xlabel('Разница побед')
    plt.ylabel('Частота')
    plt.title('Распределение разницы побед')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {"image": image_base64}

@app.get("/api/top-venues-chart")
async def top_venues_chart():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT v.Venue_Name, COUNT(*) as matches
    FROM Match m
    JOIN Venue v ON m.Venue_Id = v.Venue_Id
    GROUP BY v.Venue_Name
    ORDER BY matches DESC
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    plt.figure(figsize=(12, 6))
    plt.barh(df['Venue_Name'], df['matches'], color='coral')
    plt.xlabel('Количество матчей')
    plt.title('Самые популярные стадионы IPL')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {"image": image_base64}

@app.post("/api/train-model")
async def train_model_endpoint():
    global model, label_encoders, model_trained
    
    try:
        conn = sqlite3.connect(DB_PATH)
        matches = pd.read_sql_query("SELECT * FROM Match", conn)
        teams = pd.read_sql_query("SELECT * FROM Team", conn)
        venues = pd.read_sql_query("SELECT * FROM Venue", conn)
        conn.close()
        
        # Подготовка данных
        ml_data = matches.copy()
        team_dict = teams.set_index('Team_Id')['Team_Name'].to_dict()
        ml_data['Team1_Name'] = ml_data['Team_1'].map(team_dict)
        ml_data['Team2_Name'] = ml_data['Team_2'].map(team_dict)
        
        venue_dict = venues.set_index('Venue_Id')['Venue_Name'].to_dict()
        ml_data['Venue_Name'] = ml_data['Venue_Id'].map(venue_dict)
        ml_data['Winner_Is_Team1'] = (ml_data['Win_Type'] == 1).astype(int)
        
        ml_data_clean = ml_data.dropna(subset=['Team1_Name', 'Team2_Name', 'Venue_Name', 'Winner_Is_Team1'])
        
        label_encoders['Team1'] = LabelEncoder()
        label_encoders['Team2'] = LabelEncoder()
        label_encoders['Venue'] = LabelEncoder()
        label_encoders['Season'] = LabelEncoder()
        
        X = pd.DataFrame({
            'Team1': label_encoders['Team1'].fit_transform(ml_data_clean['Team1_Name']),
            'Team2': label_encoders['Team2'].fit_transform(ml_data_clean['Team2_Name']),
            'Venue': label_encoders['Venue'].fit_transform(ml_data_clean['Venue_Name']),
            'Season': label_encoders['Season'].fit_transform(ml_data_clean['Season_Id'].fillna(0).astype(int))
        })
        y = ml_data_clean['Winner_Is_Team1']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        model_trained = True
        
        return {
            "success": True,
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "message": f"Модель обучена с точностью {accuracy:.2%}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/predict")
async def predict_endpoint(team1: str = Form(...), team2: str = Form(...), 
                          venue: str = Form(...), season: int = Form(...)):
    global model, label_encoders, model_trained
    
    if not model_trained or model is None:
        return {"success": False, "error": "Модель не обучена"}
    
    try:
        import numpy as np
        input_data = pd.DataFrame({
            'Team1': [label_encoders['Team1'].transform([team1])[0]],
            'Team2': [label_encoders['Team2'].transform([team2])[0]],
            'Venue': [label_encoders['Venue'].transform([venue])[0]],
            'Season': [label_encoders['Season'].transform([season])[0]]
        })
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        return {
            "success": True,
            "winner": team1 if pred == 1 else team2,
            "probability": float(prob[1] if pred == 1 else prob[0]),
            "team1": team1,
            "team2": team2
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/model-info")
async def model_info():
    if not model_trained or model is None:
        return {"trained": False}
    
    # Списки для демо (заглушка)
    teams_list = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
                  'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
                  'Rajasthan Royals', 'Kings XI Punjab', 'Gujarat Lions', 'Rising Pune Supergiant']
    venues_list = ['Eden Gardens', 'Wankhede Stadium', 'M Chinnaswamy Stadium', 
                   'Feroz Shah Kotla', 'Rajiv Gandhi International Stadium']
    seasons_list = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    
    return {
        "trained": True,
        "teams": teams_list,
        "venues": venues_list,
        "seasons": seasons_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)