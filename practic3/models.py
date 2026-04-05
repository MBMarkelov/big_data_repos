import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path
import os

# Глобальные переменные для модели
model = None
label_encoders = {}
teams_list = []
venues_list = []
seasons_list = []

def train_model(matches, teams, venues):
    """Обучение модели"""
    global model, label_encoders, teams_list, venues_list, seasons_list
    
    # Подготовка данных
    ml_data = matches.copy()
    
    # Добавляем названия команд
    team_dict = teams.set_index('Team_Id')['Team_Name'].to_dict()
    ml_data['Team1_Name'] = ml_data['Team_1'].map(team_dict)
    ml_data['Team2_Name'] = ml_data['Team_2'].map(team_dict)
    
    # Добавляем стадионы
    venue_dict = venues.set_index('Venue_Id')['Venue_Name'].to_dict()
    ml_data['Venue_Name'] = ml_data['Venue_Id'].map(venue_dict)
    
    # Целевая переменная
    ml_data['Winner_Is_Team1'] = (ml_data['Win_Type'] == 1).astype(int)
    
    # Убираем строки с пропусками
    ml_data_clean = ml_data.dropna(subset=['Team1_Name', 'Team2_Name', 'Venue_Name', 'Winner_Is_Team1'])
    
    # Сохраняем списки для демо
    teams_list = sorted(ml_data_clean['Team1_Name'].unique())
    venues_list = sorted(ml_data_clean['Venue_Name'].unique())
    seasons_list = sorted(ml_data_clean['Season_Id'].dropna().unique().astype(int))
    
    # Кодирование категориальных признаков
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
    
    # Разделение и обучение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Сохранение модели
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "ipl_classifier.pkl")
    joblib.dump(label_encoders, model_dir / "label_encoders.pkl")
    
    return accuracy, len(X_train), len(X_test)

def predict_match(team1, team2, venue, season):
    """Предсказание результата матча"""
    global model, label_encoders
    
    if model is None:
        return None, None
    
    try:
        # Кодируем входные данные
        input_data = pd.DataFrame({
            'Team1': [label_encoders['Team1'].transform([team1])[0]],
            'Team2': [label_encoders['Team2'].transform([team2])[0]],
            'Venue': [label_encoders['Venue'].transform([venue])[0]],
            'Season': [label_encoders['Season'].transform([season])[0]]
        })
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        return int(pred), float(prob[1] if pred == 1 else prob[0])
    except Exception as e:
        return None, None

def get_model_info():
    """Получение информации о модели"""
    if model is None:
        return None
    return {
        'teams': teams_list,
        'venues': venues_list,
        'seasons': seasons_list
    }