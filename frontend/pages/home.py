import streamlit as st

from api_client import get_artifacts_status
from ui_utils import safe_user_error

st.title("Главная")
st.write("Веб-интерфейс для демонстрации результатов ВКР по моделям прогнозирования в профессиональном баскетболе.")

st.markdown(
    """
**Цель приложения**
- Предоставление единого интерфейса к данным и моделям ВКР.
- Демонстрация результатов прогноза очков игрока и прогноза победителя матча.
- Отображение факторов прогноза и метрик качества модели.
"""
)

st.markdown(
    """
**Используемые данные**
- Подготовленные датасеты матчей и игроков.
- Наборы признаков (`feature_sets`) для обеих задач.
- Сохраненные модели и metadata в артефактах проекта.
"""
)

st.markdown(
    """
**Интерпретация результатов**
- Прогнозы формируются по последней доступной строке контекста.
- Вероятность победы отображается только для моделей с поддержкой `predict_proba`.
- При отсутствии части признаков возвращается отдельный отчет по недостающим полям.
"""
)

ok, payload = get_artifacts_status()
st.subheader("Статус артефактов")
if not ok:
    st.error("Не удалось загрузить статус артефактов.")
    st.caption(safe_user_error(payload, "Проверьте доступность backend и конфигурацию путей."))
else:
    readiness = payload.get("readiness", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Датасеты", "Готово" if readiness.get("datasets_ready") else "Отсутствуют")
    c2.metric("Модели", "Готово" if readiness.get("models_ready") else "Частично")
    c3.metric(
        "Готовность приложения",
        "Готово" if readiness.get("app_ready_for_inference") else "Частично",
    )
    if payload.get("missing_files"):
        with st.expander("Отсутствующие файлы"):
            st.write(payload["missing_files"])
