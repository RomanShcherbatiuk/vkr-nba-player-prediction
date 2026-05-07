import streamlit as st
import pandas as pd

from api_client import (
    get_team_card,
    get_team_games,
    get_team_news,
    get_teams,
)
from ui_utils import prepare_table, safe_user_error

st.title("Команды")

search = st.text_input("Поиск команды", value="")
ok, teams_payload = get_teams(query=search or None, limit=1000)
if not ok:
    st.error("Не удалось загрузить список команд.")
    st.caption(safe_user_error(teams_payload, "Проверьте доступность данных команд в backend."))
    st.stop()

teams = teams_payload.get("items", [])
if not teams:
    st.info("Команды по заданному фильтру не найдены.")
    st.stop()

options = {t["team_name"]: t["team_id"] for t in teams}
selected_label = st.selectbox("Выберите команду", list(options.keys()))
selected_team_id = options[selected_label]

ok, card = get_team_card(selected_team_id)
if not ok:
    st.error("Не удалось загрузить карточку команды.")
    st.caption(safe_user_error(card, "Карточка команды временно недоступна."))
    st.stop()

st.subheader("Карточка команды")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Команда", card.get("team_name", "N/A"))
c2.metric("Матчей", card.get("games_count", 0))
c3.metric("Победы", card.get("wins", 0))
c4.metric("Поражения", card.get("losses", 0))
c5, c6 = st.columns(2)
c5.metric("Доля побед", card.get("win_rate", "N/A"))
c6.metric("Средние очки (proxy)", card.get("avg_points_for", "N/A"))

st.subheader("Матчи команды")
ok, games_payload = get_team_games(selected_team_id, limit=200)
if not ok:
    st.warning("Не удалось загрузить матчи команды.")
    st.caption(safe_user_error(games_payload, "Матчи команды временно недоступны."))
else:
    games = games_payload.get("items", [])
    if games:
        games_df_raw = pd.DataFrame(games)
        st.dataframe(prepare_table(games_df_raw), use_container_width=True)
        if "date" in games_df_raw.columns and "target_win" in games_df_raw.columns:
            chart_df = games_df_raw[["date", "target_win"]].dropna().copy()
            if not chart_df.empty:
                chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
                chart_df = chart_df.sort_values("date")
                st.line_chart(chart_df.set_index("date")["target_win"], use_container_width=True)
    else:
        st.info("Матчи не найдены.")

st.subheader("Новости и признаки")
ok, news_payload = get_team_news(selected_team_id, limit=100)
if not ok:
    st.warning("Не удалось загрузить данные новостей.")
    st.caption(safe_user_error(news_payload, "Новостные данные временно недоступны."))
else:
    news_items = news_payload.get("items", [])
    if news_items:
        news_df = pd.DataFrame(news_items)
        st.dataframe(prepare_table(news_df), use_container_width=True)
        if "headlines" in news_df.columns:
            with st.expander("Заголовки новостей"):
                for _, row in news_df.head(10).iterrows():
                    date_val = row.get("date") or row.get("published_at") or ""
                    st.markdown(f"**{date_val}**")
                    for title in row.get("headlines") or []:
                        st.write(f"- {title}")
    else:
        st.info(news_payload.get("message", "Новостные признаки отсутствуют."))
