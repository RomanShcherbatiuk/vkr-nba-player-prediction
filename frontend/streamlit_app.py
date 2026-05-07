import streamlit as st

from api_client import BACKEND_URL, get_artifacts_status, health_check
from ui_utils import safe_user_error


st.set_page_config(page_title="ВКР: NBA-прогнозирование", layout="wide")

st.title("Информационная система ВКР по прогнозированию в NBA")
st.write("Сервис предоставляет доступ к подготовленным данным и результатам предиктивных моделей через backend API.")
st.markdown("**Разделы приложения:**")
st.page_link("pages/home.py", label="Главная")
st.page_link("pages/players.py", label="Игроки")
st.page_link("pages/teams.py", label="Команды")
st.page_link("pages/predictions.py", label="Прогнозы")

st.subheader("Подключение к backend")
st.caption(f"Адрес backend: `{BACKEND_URL}`")
ok, payload = health_check()
if ok:
    st.success("Backend доступен.")
else:
    st.error("Backend недоступен.")
    st.caption(safe_user_error(payload, "Не удалось установить соединение с backend."))

st.subheader("Статус артефактов")
ok, status_payload = get_artifacts_status()
if not ok:
    st.warning("Не удалось получить статус артефактов.")
    st.caption(safe_user_error(status_payload, "Проверьте доступность backend и конфигурацию окружения."))
else:
    readiness = status_payload.get("readiness", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Датасеты", "Готово" if readiness.get("datasets_ready") else "Отсутствуют")
    c2.metric("Модели", "Готово" if readiness.get("models_ready") else "Частично")
    c3.metric(
        "Готовность inference",
        "Готово" if readiness.get("app_ready_for_inference") else "Частично",
    )
