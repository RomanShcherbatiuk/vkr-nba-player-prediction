# ВКР: прогнозирование показателей NBA

Проект содержит Streamlit-приложение и FastAPI backend для демонстрации моделей прогнозирования в NBA.

## Структура

```text
.
├── README.md
├── requirements.txt
├── docker-compose.yml
├── backend/
│   └── app/
├── frontend/
│   ├── streamlit_app.py
│   └── pages/
├── data/
└── models/
```

## Необходимые данные и модели

Папки `data/` и `models/` не хранятся в Git и загружаются отдельно.

Источник артефактов:

- Google Drive: https://drive.google.com/drive/folders/1gj-G0TYmQqHZbsXFT5oTu5N09FY_mc2I?usp=sharing

После скачивания разместите содержимое папок в:

- `data/`
- `models/`

Для запуска используются следующие файлы:

- `data/match_prediction_dataset.csv`
- `data/match_prediction_feature_sets.json`
- `data/player_points_prediction_dataset.csv`
- `data/player_points_feature_sets.json`
- `data/nba_news_daily_features.csv`
- `data/nba_news_clean.parquet`
- `models/match_winner_model.joblib`
- `models/match_winner_model_metadata.json`
- `models/player_points_model.joblib`
- `models/player_points_model_metadata.json`

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

В первом терминале:

```bash
PROJECT_ROOT=$(pwd) uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Во втором терминале:

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Также доступен запуск через Docker:

```bash
docker compose up --build
```

Backend доступен по адресу `http://localhost:8000`, Streamlit-приложение - по адресу `http://localhost:8501`.

## Основные файлы

- `frontend/streamlit_app.py` - главная страница Streamlit.
- `frontend/pages/` - разделы приложения.
- `frontend/api_client.py` - клиент backend API.
- `backend/app/main.py` - точка входа FastAPI.
- `backend/app/services/` - бизнес-логика.
- `backend/app/repositories/` - загрузка данных и моделей.

## Поддерживаемые прогнозы

- прогноз очков игрока;
- прогноз победителя матча.
