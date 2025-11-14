# main_fastapi.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional

# 自作モジュールのインポート
from database import get_user, log_interaction
from esa_connector import fetch_esa_documents

app = FastAPI()


# Pydantic モデル
class User(BaseModel):
    username: str
    password: str


class QuestionRequest(BaseModel):
    question: str
    user_id: str


class LoggedInUser(BaseModel):
    id: int
    username: str


# --- RAG システムの初期化と構築 ---

# 1. モデル設定
# Gated Repositoryのため、Hugging Face CLIで認証が必要です
MODEL_NAME = "elyza/Llama-3-ELYZA-JP-8B"

# 2. 埋め込みモデル初期化 (キャッシュで使用)
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
print("✅ 埋め込みモデルの初期化完了。")

# 3. esaドキュメントの取得とFAISSインデックスのロード（キャッシュ利用）
# fetch_esa_documentsは、キャッシュからロードするか、APIから新規取得/保存します
documents_with_metadata, index = fetch_esa_documents(embedding_model=embedding_model)

documents = [d[0] for d in documents_with_metadata]
metadatas = [d[1] for d in documents_with_metadata]

# 4. 生成モデル（LLM）の初期化
print(f"⏳ 生成モデル ({MODEL_NAME}) をロード中...")

# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# モデルのロード (GPU利用設定の例: Macではmps, Linuxではcuda)
# 7Bモデルはメモリ消費が大きいため、メモリが少ない場合は load_in_8bit=True を検討
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype="auto"  # 自動で適切な精度を選択
)

# Pipelineの構築
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device_map="auto" を指定しているため、device=-1 は不要
)

print(f"✅ 生成モデル ({MODEL_NAME}) のロード完了。")

# --- RAG 検索・生成ロジック ---


def retrieve(query, top_k=2) -> List[Dict]:
    """FAISSインデックスから関連ドキュメントを検索する"""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, i in enumerate(indices[0]):
        # ドキュメントのテキストとメタデータを取得
        if i >= 0:  # 正常なインデックスのみ処理
            results.append(
                {
                    "context": documents[i],
                    "source": metadatas[i]["source"],
                    "distance": distances[0][idx],
                }
            )
    return results


def generate_answer(question: str, relevant_docs: List[Dict]) -> str:
    """検索結果と質問を基にLLMで回答を生成する"""
    if (
        not relevant_docs
        or relevant_docs[0]["context"] == "esaから情報が取得されていません。"
    ):
        return "esa wikiから関連情報を見つけられませんでした。"

    contexts = "\n---\n".join([doc["context"] for doc in relevant_docs])

    # Instructモデルに合わせたプロンプト形式
    prompt = f"""
[INST]
以下はesa wikiからの参考情報です。この情報のみを使用して、次の質問に簡潔に日本語で答えてください。情報が不足している場合は、「情報が見つかりません。」と答えてください。
参考情報:
{contexts}

質問: {question}
[/INST]

回答:"""

    # 応答生成
    response = generator(
        prompt,
        max_new_tokens=256,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # 回答の多様性を持たせる
        truncation=True,
    )

    generated_text = response[0]["generated_text"]

    # プロンプトとインストラクションタグを削除して回答のみを抽出
    # Llama系のInstructモデルでは、回答が [/INST] の直後から始まることが多い
    try:
        # [/INST] の後の部分を抽出
        answer_start = generated_text.rfind("[/INST]")
        if answer_start != -1:
            answer = generated_text[answer_start + len("[/INST]") :].strip()
        else:
            answer = generated_text  # エラー時
    except:
        answer = generated_text

    return answer


# --- FastAPI エンドポイント ---


@app.post("/login", response_model=LoggedInUser)
async def login(user: User):
    """ユーザー認証を行う（ダミー）"""
    db_user = get_user(user.username, user.password)
    if db_user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return db_user


@app.post("/rag/ask")
async def rag_ask(req: QuestionRequest):
    """質問を受け付け、RAGを実行し、ログを記録する"""
    try:
        # 1. 検索 (Retrieval)
        relevant_docs = retrieve(req.question, top_k=3)

        # 2. 生成 (Generation)
        answer = generate_answer(req.question, relevant_docs)
        sources = [
            {"text": doc["context"], "source": doc["source"]}
            for doc in relevant_docs
            if doc.get("context") != "esaから情報が取得されていません。"
        ]

        # 3. ロギング
        log_interaction(req.user_id, req.question, answer, sources)

        return {"question": req.question, "answer": answer, "sources": sources}

    except Exception as e:
        print(f"Error in RAG process: {e}")
        # ログ記録（エラーログ）
        # log_error(req.user_id, req.question, str(e))
        raise HTTPException(
            status_code=500,
            detail="RAG処理中にエラーが発生しました。詳細はサーバーログを確認してください。",
        )


# 実行方法:
# uvicorn main_fastapi:app --reload
