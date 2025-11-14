# frontend_streamlit.py
import streamlit as st
import requests
import json

# FastAPIのURL
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="esa RAG Q&A システム", layout="wide")

st.title("💡 esa Q&A システム")

# --- 認証セクション ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None


def handle_login(username, password):
    """ログインAPIを呼び出す"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/login", json={"username": username, "password": password}
        )
        if response.status_code == 200:
            user_data = response.json()
            st.session_state.logged_in = True
            st.session_state.user_id = user_data.get("id")
            st.session_state.username = user_data.get("username")
            st.success(f"ログイン成功: {st.session_state.username}さん")
            st.rerun()
        else:
            st.error("ログイン情報が正しくありません。")
    except Exception as e:
        st.error(f"接続エラー: {e}")


def handle_logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()


if not st.session_state.logged_in:
    st.subheader("ログイン")
    with st.form("login_form"):
        username = st.text_input("ユーザー名 (例: testuser)")
        password = st.text_input("パスワード (例: password123)", type="password")
        submitted = st.form_submit_button("ログイン")
        if submitted:
            handle_login(username, password)
else:
    st.sidebar.success(f"ようこそ、{st.session_state.username}さん")
    st.sidebar.button("ログアウト", on_click=handle_logout)

    # --- Q&Aセクション ---
    st.subheader("esa wiki 質問応答")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area(
        "esa wiki の情報について質問を入力してください:", height=100
    )

    if st.button("質問する"):
        if question:
            with st.spinner("RAGシステムが回答を生成中です..."):
                try:
                    payload = {
                        "question": question,
                        "user_id": str(st.session_state.user_id),
                    }
                    # FastAPIのRAGエンドポイントを呼び出し
                    response = requests.post(f"{API_BASE_URL}/rag/ask", json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.history.append(
                            {
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                            }
                        )
                    else:
                        st.error(
                            f"APIエラーが発生しました: {response.status_code} - {response.text}"
                        )
                except Exception as e:
                    st.error(f"API接続エラー: {e}")

    # --- 履歴表示セクション ---
    if st.session_state.history:
        st.subheader("質問履歴")
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q{len(st.session_state.history) - i}: {item['question']}**")
            st.info(f"**A:** {item['answer']}")

            with st.expander("参照された esa ドキュメント (ソース)"):
                if item["sources"]:
                    for src in item["sources"]:
                        st.markdown(f"**タイトル:** `{src['source']}`")
                        st.markdown(f"> {src['text']}")
                        st.markdown("---")
                else:
                    st.markdown("関連情報は見つかりませんでした。")
            st.markdown("---")
