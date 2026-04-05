# app.py
"""
Streamlit приложение для анализа данных IPL
Практическая работа 2
"""

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="IPL Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏏 Indian Premier League (IPL) Analytics Dashboard")
st.markdown("---")

@st.cache_resource
def load_data():
    """Загрузка данных из SQLite"""
    db_path = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"

    conn = sqlite3.connect(str(db_path))
    
    # Загрузка таблиц
    matches = pd.read_sql_query("SELECT * FROM Match", conn)
    teams = pd.read_sql_query("SELECT * FROM Team", conn)
    seasons = pd.read_sql_query("SELECT * FROM Season", conn)
    venues = pd.read_sql_query("SELECT * FROM Venue", conn)
    
    conn.close()
    
    return matches, teams, seasons, venues

with st.spinner("Загрузка данных..."):
    matches, teams, seasons, venues = load_data()

st.success(f"✅ Загружено {len(matches)} матчей, {len(teams)} команд, {len(seasons)} сезонов")

st.sidebar.title("🏏 IPL Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Навигация",
    ["📋 Обзор проекта", "📊 EDA Анализ", "🤖 ML Модель", "📈 Результаты запросов"]
)

if page == "📋 Обзор проекта":
    st.header("📋 Обзор проекта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Цель проекта")
        st.write("""
        Создание интерактивной веб-витрины для анализа данных 
        **Indian Premier League (IPL)** — самой популярной крикетной лиги в мире.
        """)
        
        st.subheader("📊 Источник данных")
        st.write("""
        - **Матчи:** 577 записей (2008-2019)
        - **Команды:** 13 участников
        - **Стадионы:** 35 площадок
        - **Игроки:** 469 спортсменов
        """)
    
    with col2:
        st.subheader("🔧 Используемые технологии")
        st.write("""
        - **Python** + **Streamlit** — веб-фреймворк
        - **SQLite** — хранение данных
        - **Pandas** — обработка данных
        - **Matplotlib/Seaborn** — визуализация
        - **Scikit-learn** — машинное обучение
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Ключевые метрики")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("Всего матчей", len(matches))
    with col4:
        st.metric("Команд", len(teams))
    with col5:
        st.metric("Сезонов", len(seasons))
    with col6:
        avg_runs = matches['Win_Margin'].mean()
        st.metric("Средний разрыв", f"{avg_runs:.1f}")

elif page == "📊 EDA Анализ":
    st.header("📊 Исследовательский анализ данных (EDA)")
    
    matches_with_seasons = matches.merge(seasons, on='Season_Id')
    matches_with_venues = matches.merge(venues, on='Venue_Id')
    
    st.subheader("📅 Распределение матчей по сезонам")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    season_counts = matches_with_seasons.groupby('Season_Year').size()
    ax1.bar(season_counts.index.astype(str), season_counts.values, color='steelblue')
    ax1.set_xlabel('Сезон')
    ax1.set_ylabel('Количество матчей')
    ax1.set_title('Матчи IPL по годам')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    st.subheader("🏆 Распределение разницы побед")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(matches['Win_Margin'].dropna(), bins=30, color='forestgreen', edgecolor='black')
    ax2.set_xlabel('Разница побед')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение разницы побед')
    st.pyplot(fig2)
    
    st.subheader("🏟️ Топ-10 стадионов по количеству матчей")
    venue_counts = matches_with_venues.groupby('Venue_Name').size().sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.barh(venue_counts.index, venue_counts.values, color='coral')
    ax3.set_xlabel('Количество матчей')
    ax3.set_title('Самые популярные стадионы IPL')
    st.pyplot(fig3)
    
    st.subheader("📊 Статистика по матчам")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Типы побед:**")
        win_types = matches['Win_Type'].value_counts()
        win_labels = {1: 'По ранам', 2: 'По калиткам'}
        win_df = pd.DataFrame({'Тип': [win_labels.get(x, x) for x in win_types.index], 'Кол-во': win_types.values})
        st.dataframe(win_df)
    
    with col2:
        st.write("**Решения после тосса:**")
        toss_decisions = matches['Toss_Decide'].value_counts()
        toss_labels = {1: 'Field', 2: 'Bat'}
        toss_df = pd.DataFrame({'Решение': [toss_labels.get(x, x) for x in toss_decisions.index], 'Кол-во': toss_decisions.values})
        st.dataframe(toss_df)

elif page == "🤖 ML Модель":
    st.header("🤖 Машинное обучение - Предсказание победителя")
    
    st.info("""
    **Модель:** Логистическая регрессия для предсказания победителя матча
    **Признаки:** 
    - Команда 1 и Команда 2
    - Стадион
    - Решение после тосса
    - Сезон
    """)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    import os
    
    st.subheader("🎯 Обучение модели")
    
    ml_data = matches.copy()
    
    team_dict = teams.set_index('Team_Id')['Team_Name'].to_dict()
    ml_data['Team1_Name'] = ml_data['Team_1'].map(team_dict)
    ml_data['Team2_Name'] = ml_data['Team_2'].map(team_dict)
    
    venue_dict = venues.set_index('Venue_Id')['Venue_Name'].to_dict()
    ml_data['Venue_Name'] = ml_data['Venue_Id'].map(venue_dict)
    
    ml_data['Winner_Is_Team1'] = (ml_data['Win_Type'] == 1).astype(int)
    
    ml_data_clean = ml_data.dropna(subset=['Team1_Name', 'Team2_Name', 'Venue_Name', 'Winner_Is_Team1'])
    
    le_team1 = LabelEncoder()
    le_team2 = LabelEncoder()
    le_venue = LabelEncoder()
    le_season = LabelEncoder()
    
    X = pd.DataFrame({
        'Team1': le_team1.fit_transform(ml_data_clean['Team1_Name']),
        'Team2': le_team2.fit_transform(ml_data_clean['Team2_Name']),
        'Venue': le_venue.fit_transform(ml_data_clean['Venue_Name']),
        'Season': le_season.fit_transform(ml_data_clean['Season_Id'].fillna(0).astype(int))
    })
    y = ml_data_clean['Winner_Is_Team1']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"**Размер обучающей выборки:** {len(X_train)} матчей")
    st.write(f"**Размер тестовой выборки:** {len(X_test)} матчей")
    
    with st.spinner("Обучение модели..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    st.success(f"✅ Точность модели: **{accuracy:.2%}**")
    
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ipl_classifier.pkl"
    joblib.dump(model, model_path)
    st.write(f"💾 Модель сохранена в `{model_path}`")
    
    # Демо предсказания
    st.subheader("🔮 Демо: Предсказание результата матча")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_list = sorted(ml_data_clean['Team1_Name'].unique())
        team1 = st.selectbox("Команда 1", team_list)
        team2 = st.selectbox("Команда 2", [t for t in team_list if t != team1])
    
    with col2:
        venue_list = sorted(ml_data_clean['Venue_Name'].unique())
        venue = st.selectbox("Стадион", venue_list)
        season = st.selectbox("Сезон", sorted(ml_data_clean['Season_Id'].dropna().unique().astype(int)))
    
    if st.button("Предсказать победителя"):
        input_data = pd.DataFrame({
            'Team1': [le_team1.transform([team1])[0]],
            'Team2': [le_team2.transform([team2])[0]],
            'Venue': [le_venue.transform([venue])[0]],
            'Season': [le_season.transform([season])[0]]
        })
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        if pred == 1:
            st.success(f"🏆 **{team1}** победит с вероятностью {prob[1]:.1%}")
        else:
            st.success(f"🏆 **{team2}** победит с вероятностью {prob[0]:.1%}")

else:
    st.header("📈 Результаты SQL запросов (из Практической работы 1)")
    
    st.subheader("1. Статистика по сезонам")
    query1 = """
    SELECT 
        s.Season_Year,
        COUNT(*) as matches,
        ROUND(AVG(m.Win_Margin), 1) as avg_margin
    FROM Match m
    JOIN Season s ON m.Season_Id = s.Season_Id
    GROUP BY s.Season_Year
    ORDER BY s.Season_Year DESC
    """
    db_path = r"C:\Users\MB_Markelov_Nout\Documents\GitHub\big_data_repos\lab1\data\daatabase.sqlite"

    df1 = pd.read_sql_query(query1, sqlite3.connect(db_path))
    st.dataframe(df1, use_container_width=True)

    st.subheader("2. Топ-5 стадионов по матчам")
    query2 = """
    SELECT 
        v.Venue_Name,
        COUNT(*) as matches
    FROM Match m
    JOIN Venue v ON m.Venue_Id = v.Venue_Id
    GROUP BY v.Venue_Name
    ORDER BY matches DESC
    LIMIT 5
    """

    df2 = pd.read_sql_query(query2, sqlite3.connect(db_path))
    st.dataframe(df2, use_container_width=True)
    
    st.subheader("3. Топ-5 команд по победам")
    query3 = """
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

    df3 = pd.read_sql_query(query3, sqlite3.connect(db_path))
    st.dataframe(df3, use_container_width=True)
    
    st.subheader("4. Влияние тосса")
    query4 = """
    SELECT 
        td.Toss_Name,
        COUNT(*) as total_matches
    FROM Match m
    JOIN Toss_Decision td ON m.Toss_Decide = td.Toss_Id
    GROUP BY td.Toss_Name
    """
    df4 = pd.read_sql_query(query4, sqlite3.connect(db_path))
    st.dataframe(df4, use_container_width=True)

st.markdown("---")
st.caption("© IPL Analytics Dashboard | Практическая работа 2")