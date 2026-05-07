import streamlit as st
import pandas as pd

from api_client import (
    get_player_card,
    get_player_features,
    get_player_games,
    get_players,
)
from ui_utils import prepare_table, safe_user_error

st.title("Игроки")

search = st.text_input("Поиск игрока", value="")
ok, players_payload = get_players(query=search or None, limit=1000)
if not ok:
    st.error("Не удалось загрузить список игроков.")
    st.caption(safe_user_error(players_payload, "Проверьте доступность данных игроков в backend."))
    st.stop()

players = players_payload.get("items", [])
if not players:
    st.info("Игроки по заданному фильтру не найдены.")
    st.stop()

options = {f"{p['player_name']} ({p.get('team_name') or 'N/A'})": p["player_id"] for p in players}
selected_label = st.selectbox("Выберите игрока", list(options.keys()))
selected_player_id = options[selected_label]

ok, card = get_player_card(selected_player_id)
if not ok:
    st.error("Не удалось загрузить карточку игрока.")
    st.caption(safe_user_error(card, "Карточка игрока временно недоступна."))
    st.stop()

st.subheader("Карточка игрока")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Игрок", card.get("player_name", "N/A"))
c2.metric("Матчей", card.get("games_count", 0))
c3.metric("Средние очки", card.get("avg_points", "N/A"))
c4.metric("Средние минуты", card.get("avg_minutes", "N/A"))
st.caption(f"Команда: {card.get('team_name') or 'N/A'}")
if card.get("seasons"):
    st.caption(f"Сезоны: {', '.join(card['seasons'][:10])}{' ...' if len(card['seasons']) > 10 else ''}")

st.subheader("Матчи игрока")
season_options = ["Все"] + [str(s) for s in card.get("seasons", [])]
selected_season = st.selectbox("Фильтр сезона", season_options)
season_param = None if selected_season == "Все" else selected_season

ok, games_payload = get_player_games(selected_player_id, season=season_param, limit=200)
if not ok:
    st.warning("Не удалось загрузить матчи игрока.")
    st.caption(safe_user_error(games_payload, "Матчи игрока временно недоступны."))
else:
    games = games_payload.get("items", [])
    if games:
        games_df_raw = pd.DataFrame(games)
        st.dataframe(prepare_table(games_df_raw), use_container_width=True)
        if "date" in games_df_raw.columns and "points" in games_df_raw.columns:
            chart_df = games_df_raw[["date", "points"]].dropna().copy()
            if not chart_df.empty:
                chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
                chart_df = chart_df.sort_values("date")
                st.line_chart(chart_df.set_index("date")["points"], use_container_width=True)
    else:
        st.info("Матчи для выбранного фильтра не найдены.")

with st.expander("Признаки игрока"):
    ok, features_payload = get_player_features(selected_player_id)
    if not ok:
        st.caption("Признаки игрока временно недоступны.")
    else:
        features = features_payload.get("features", [])
        if features:
            st.dataframe(prepare_table(pd.DataFrame(features)), use_container_width=True)
        else:
            st.info("Список признаков недоступен.")
