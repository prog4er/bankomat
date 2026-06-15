import json
import requests
import streamlit as st

st.set_page_config(page_title="ATM Simulator (Учебный)", layout="centered")

DEFAULT_BASE_URL = "http://localhost:8000"

if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_BASE_URL
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.title("Учебный симулятор банкомата")

with st.sidebar:
    st.header("Настройки")
    st.session_state.base_url = st.text_input("Base URL API", st.session_state.base_url)

base = st.session_state.base_url.rstrip("/")


def api(method: str, path: str, payload=None):
    url = f"{base}{path}"
    if method == "GET":
        r = requests.get(url, timeout=10)
    else:
        r = requests.post(url, json=payload or {}, timeout=10)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = {"detail": r.text}
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")
    return r.json()


def show_json(obj):
    st.code(json.dumps(obj, ensure_ascii=False, indent=2), language="json")


if st.session_state.session_id is None:
    st.subheader("Вход: карта + PIN (учебно)")
    card = st.text_input("card_token", "CARD-0001")
    pin = st.text_input("PIN", type="password", value="1234")
    ttl = st.caption("Подсказка: демо‑карты: CARD-0001 (PIN 1234), CARD-0002 (PIN 4321).")

    if st.button("Войти"):
        try:
            resp = api("POST", "/api/v1/auth/start", {"card_token": card, "pin": pin})
            st.session_state.session_id = resp["session_id"]
            st.success(resp.get("message", "OK"))
            show_json(resp)
            st.rerun()
        except Exception as e:
            st.error(str(e))

else:
    sid = st.session_state.session_id

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Сессия: {sid[:8]}…")
    with col2:
        if st.button("Отмена"):
            try:
                resp = api("POST", f"/api/v1/session/{sid}/cancel")
                show_json(resp)
            finally:
                st.session_state.session_id = None
                st.rerun()

    # инфо о сессии + тайм‑аут
    try:
        info = api("GET", f"/api/v1/session/{sid}")
        st.info(f"state={info['state']}, seconds_left≈{info.get('seconds_left')}")
    except Exception as e:
        st.warning(f"Сессия недоступна: {e}")
        st.session_state.session_id = None
        st.rerun()

    st.markdown("### Операции")
    op = st.radio("Выберите операцию", ["Баланс", "Снятие", "Пополнение", "Перевод"])

    if op == "Баланс":
        if st.button("Запросить баланс"):
            try:
                resp = api("POST", f"/api/v1/session/{sid}/balance")
                st.success("Готово")
                show_json(resp)
            except Exception as e:
                st.error(str(e))

    elif op in ("Снятие", "Пополнение"):
        amount = st.number_input("Сумма (копейки/центы)", min_value=1, value=1000, step=100)
        if st.button("Выполнить"):
            try:
                path = "withdraw" if op == "Снятие" else "deposit"
                resp = api("POST", f"/api/v1/session/{sid}/{path}", {"amount_cents": int(amount)})
                st.success(resp.get("message", "OK"))
                show_json(resp)
            except Exception as e:
                st.error(str(e))

    elif op == "Перевод":
        target = st.text_input("target_card_token", "CARD-0002")
        amount = st.number_input("Сумма (копейки/центы)", min_value=1, value=500, step=100)
        if st.button("Перевести"):
            try:
                resp = api("POST", f"/api/v1/session/{sid}/transfer", {"target_card_token": target, "amount_cents": int(amount)})
                st.success(resp.get("message", "OK"))
                show_json(resp)
            except Exception as e:
                st.error(str(e))

    st.markdown("### Выход")
    if st.button("Завершить (выйти)"):
        try:
            api("POST", f"/api/v1/session/{sid}/cancel")
        finally:
            st.session_state.session_id = None
            st.rerun()
