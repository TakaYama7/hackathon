import requests
import re
import json
import time
import os
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# --- 1. 設定値（ユーザーごとに置き換えてください） ---
# ⚠️ チーム名をサブドメイン部分のみに修正してください
ESA_TEAM_NAME = "cs18a"
ESA_ACCESS_TOKEN = "Id6WLrpYfGhF8-l0MsuMI--55xUwn3JfTYmzXVZWpHo"

# URLはf-stringで正しく構築されます
ESA_API_BASE_URL = f"https://api.esa.io/v1/teams/{ESA_TEAM_NAME}"
HEADERS = {
    "Authorization": f"Bearer {ESA_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}

# --- キャッシュ設定 ---
CACHE_FILE = "esa_data_cache.json"
FAISS_INDEX_FILE = "esa_faiss_index.bin"
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 1日 = 86400秒

# --- ユーティリティ関数 ---


def clean_markdown(markdown_text: str) -> str:
    """
    Markdown 記法を簡易的に除去し、プレーンテキストを抽出する。
    """
    # 見出し、リンク、画像を削除
    text = re.sub(r"#{1,6}\s?", "", markdown_text)  # 見出し
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # リンクと画像
    text = re.sub(
        r"(\*\*|__|~~|\*|_|`)", "", text
    )  # 太字、イタリック、インラインコード
    text = re.sub(r"^\s*[-*+]\s", "", text, flags=re.MULTILINE)  # リスト
    text = re.sub(r"\n{2,}", "\n", text)  # 連続する改行を一つに
    return text.strip()


def simple_text_splitter(text: str, chunk_size: int = 500) -> List[str]:
    """
    テキストを指定サイズで簡易的にチャンク（分割）する。
    """
    chunks = []
    # テキストが空でなければ分割
    if text:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])
    return chunks


# --- キャッシュ機能 ---


def check_cache_validity() -> bool:
    """キャッシュファイルが存在し、有効期限内かチェックする"""
    if not os.path.exists(CACHE_FILE) or not os.path.exists(FAISS_INDEX_FILE):
        return False

    file_mtime = os.path.getmtime(CACHE_FILE)
    time_since_last_update = time.time() - file_mtime

    return time_since_last_update < CACHE_EXPIRY_SECONDS


def load_from_cache(
    embedding_model: SentenceTransformer,
) -> Tuple[List[Tuple[str, Dict]], faiss.Index]:
    """キャッシュファイルからデータとFAISSインデックスを読み込む"""
    print(f"✅ キャッシュが有効です。データをロードします。")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents_with_metadata = [(item["chunk"], item["metadata"]) for item in data]

    # FAISSインデックスのロード
    index = faiss.read_index(FAISS_INDEX_FILE)

    return documents_with_metadata, index


def save_to_cache(documents_with_metadata: List[Tuple[str, Dict]], index: faiss.Index):
    """取得したデータをキャッシュファイルとFAISSインデックスとして保存する"""

    # 1. JSONキャッシュの保存
    cache_data = []
    for chunk, metadata in documents_with_metadata:
        cache_data.append({"chunk": chunk, "metadata": metadata})

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

    # 2. FAISSインデックスの保存
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"💾 記事データとFAISSインデックスをキャッシュに保存しました。")


# --- 記事の取得と処理のメイン関数 ---


def fetch_esa_documents(
    embedding_model: SentenceTransformer,
) -> Tuple[List[Tuple[str, Dict]], faiss.Index]:
    """
    esa APIから記事を取得し、RAGに必要な形式に処理する（キャッシュチェック付き）
    """

    # 1. キャッシュのチェック
    if check_cache_validity():
        try:
            return load_from_cache(embedding_model)
        except Exception as e:
            print(
                f"⚠️ キャッシュのロード中にエラーが発生しました: {e}。APIから再取得します。"
            )

    # --- 2. キャッシュが無効なため、APIから新規取得 ---
    all_documents = []
    page = 1

    while True:
        try:
            list_url = f"{ESA_API_BASE_URL}/posts"
            params = {"per_page": 100, "page": page, "include": "tags,category"}
            response = requests.get(list_url, headers=HEADERS, params=params)

            if response.status_code != 200:
                print(f"--- 致命的な API エラー ---")
                print(f"ステータスコード: {response.status_code}")
                print(
                    f"エラーメッセージ: {response.json().get('error', response.text)}"
                )
                # 認証やURLのエラーのため、ここで取得を中止
                break

            posts_data = response.json()
            posts = posts_data.get("posts", [])

            if not posts:
                break

            print(f"--- ページ {page}: {len(posts)} 件の記事を処理中 ---")

            for post in posts:
                post_number = post["number"]

                # 記事詳細を取得し、Markdown本文を取得
                detail_url = f"{ESA_API_BASE_URL}/posts/{post_number}"
                detail_response = requests.get(detail_url, headers=HEADERS)
                detail_response.raise_for_status()
                post_detail = detail_response.json()

                body_md = post_detail.get("body_md", "")

                # Markdownをクリーニングし、チャンクに分割
                plain_text = clean_markdown(body_md)
                chunks = simple_text_splitter(plain_text)

                # RAG用にドキュメントリストに追加
                for chunk in chunks:
                    metadata = {
                        "source": post["url"],
                        "title": post["full_name"],
                        "category": post.get("category"),
                        "tags": post.get("tags"),
                    }
                    all_documents.append((chunk, metadata))

            page += 1
            if posts_data.get("next_page") is None:
                break

        except requests.exceptions.RequestException as e:
            print(f"API リクエストエラー: {e}")
            break
        except Exception as e:
            print(f"予期せぬエラー: {e}")
            break

    # --- 3. 取得したデータのベクトル化とインデックス作成 ---
    if not all_documents:
        print(
            "⚠️ 警告: esa APIからドキュメントが取得できませんでした。ダミーデータを返します。"
        )
        # エラー回避のためのダミーデータ作成
        dummy_chunk = "esaから情報が取得されていません。"
        dummy_embedding = embedding_model.encode([dummy_chunk])
        dimension = dummy_embedding.shape[1]
        dummy_index = faiss.IndexFlatL2(dimension)
        dummy_index.add(dummy_embedding)
        return [(dummy_chunk, {"source": "No Data"})], dummy_index

    documents = [d[0] for d in all_documents]
    doc_embeddings = embedding_model.encode(documents)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # 4. 新しいキャッシュの保存
    save_to_cache(all_documents, index)

    print(
        f"--- 処理完了: {len(all_documents)} 個のドキュメントチャンクを作成しました ---"
    )
    return all_documents, index
