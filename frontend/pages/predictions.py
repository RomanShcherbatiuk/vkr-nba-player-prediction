import streamlit as st
import pandas as pd

from api_client import (
    get_matchup_prediction,
    get_player_prediction,
    get_players,
    get_teams,
)
from ui_utils import safe_user_error

st.title("Прогнозы")
st.write("Выберите тип прогноза и заполните форму.")

prediction_type = st.radio(
    "Тип прогноза",
    ["Прогноз очков игрока", "Прогноз победителя матча"],
    horizontal=True,
)

if prediction_type == "Прогноз очков игрока":
    st.subheader("Прогноз очков игрока")
    ok, players_payload = get_players(limit=300)
    if not ok or not players_payload.get("items"):
        st.warning("Список игроков недоступен.")
    else:
        players = players_payload["items"]
        options = {p["player_name"]: p["player_id"] for p in players}
        selected_player = st.selectbox("Игрок", list(options.keys()), key="pred_player")
        if st.button("Сформировать прогноз", key="btn_player_pred"):
            ok, prediction = get_player_prediction(options[selected_player])
            if not ok:
                st.error("Не удалось получить прогноз очков игрока.")
                st.caption(safe_user_error(prediction, "Сервис прогноза временно недоступен."))
            elif prediction.get("status") != "ok":
                st.error("Прогноз временно недоступен.")
            else:
                st.metric("Прогноз очков", prediction.get("predicted_points", "Н/Д"))

else:
    st.subheader("Прогноз победителя матча")
    ok, teams_payload = get_teams(limit=300)
    if not ok or not teams_payload.get("items"):
        st.warning("Список команд недоступен.")
    else:
        teams = teams_payload["items"]
        options = {t["team_name"]: t["team_id"] for t in teams}
        team_names = list(options.keys())

        selected_team_1 = st.selectbox("Первая команда", team_names, key="pred_team_1")
        default_index = 1 if len(team_names) > 1 else 0
        selected_team_2 = st.selectbox("Вторая команда", team_names, index=default_index, key="pred_team_2")
        season = st.text_input("Сезон (при необходимости)", value="", key="pred_match_season")
        use_match_date = st.checkbox("Указать дату матча", value=False, key="pred_use_match_date")
        match_date = st.date_input("Дата матча", format="YYYY-MM-DD", key="pred_match_date") if use_match_date else None

        col_team_1, col_team_2 = st.columns(2)
        with col_team_1:
            row_1 = next((x for x in teams if x["team_name"] == selected_team_1), {})
            st.markdown("**Первая команда**")
            st.write(f"Название: {selected_team_1}")
            st.write(f"Матчей: {row_1.get('games_count', 'Н/Д')}")
            st.write(f"Побед: {row_1.get('wins', 'Н/Д')}")
            st.write(f"Поражений: {row_1.get('losses', 'Н/Д')}")
            st.write(f"Доля побед: {row_1.get('win_rate', 'Н/Д')}")

        with col_team_2:
            row_2 = next((x for x in teams if x["team_name"] == selected_team_2), {})
            st.markdown("**Вторая команда**")
            st.write(f"Название: {selected_team_2}")
            st.write(f"Матчей: {row_2.get('games_count', 'Н/Д')}")
            st.write(f"Побед: {row_2.get('wins', 'Н/Д')}")
            st.write(f"Поражений: {row_2.get('losses', 'Н/Д')}")
            st.write(f"Доля побед: {row_2.get('win_rate', 'Н/Д')}")

        if st.button("Сформировать прогноз", key="btn_matchup_pred"):
            if selected_team_1 == selected_team_2:
                st.error("Выберите две разные команды.")
            else:
                ok, prediction = get_matchup_prediction(
                    team_1_id=options[selected_team_1],
                    team_2_id=options[selected_team_2],
                    season=season.strip() or None,
                    match_date=None if match_date is None else str(match_date),
                )
                if not ok:
                    st.error("Не удалось получить прогноз победителя матча.")
                    st.caption(safe_user_error(prediction, "Сервис прогноза временно недоступен."))
                elif prediction.get("status") != "ok":
                    st.error("Прогноз временно недоступен.")
                else:
                    p1 = prediction.get("probability_team_1")
                    p2 = prediction.get("probability_team_2")
                    winner = prediction.get("predicted_winner", "Н/Д")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Прогнозируемый победитель", winner)
                    c2.metric(
                        f"Вероятность победы: {selected_team_1}",
                        f"{float(p1):.1%}" if isinstance(p1, (int, float)) else "Н/Д",
                    )
                    c3.metric(
                        f"Вероятность победы: {selected_team_2}",
                        f"{float(p2):.1%}" if isinstance(p2, (int, float)) else "Н/Д",
                    )
                    st.caption(prediction.get("summary_ru", "Краткое пояснение недоступно."))

                    st.markdown("**Ключевые признаки, влияющие на прогноз**")
                    key_features = prediction.get("key_model_features", [])
                    if key_features:
                        st.write(", ".join([str(x) for x in key_features]))
                    else:
                        st.caption("Список ключевых признаков недоступен.")

                    st.markdown("**Разницы между ключевыми показателями команд**")
                    key_diffs = prediction.get("key_feature_differences", [])
                    if key_diffs:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Признак": x.get("feature_name"),
                                        f"{selected_team_1}": x.get("team_1_value"),
                                        f"{selected_team_2}": x.get("team_2_value"),
                                        "Разница (1 - 2)": x.get("diff"),
                                    }
                                    for x in key_diffs
                                ]
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Разницы показателей недоступны.")
