import streamlit as st
import requests
import numpy as np
import json
import os

from utils.document_loader import extract_text_with_pages
from utils.text_chunker import chunk_text
from utils.embeddings import generate_embeddings, model
from utils.vector_store import VectorStore

st.set_page_config(page_title="AnswerAI", layout="wide")

# -----------------------------
# FILE PATHS
# -----------------------------
USER_DB = "users.json"
CHAT_DB = "chat_history.json"

# -----------------------------
# USER DB
# -----------------------------
def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# -----------------------------
# CHAT DB
# -----------------------------
def load_chat_history():
    if not os.path.exists(CHAT_DB):
        return {}
    with open(CHAT_DB, "r") as f:
        return json.load(f)

def save_chat_history(data):
    with open(CHAT_DB, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# AUTH SYSTEM
# -----------------------------
def login_signup():
    st.title("🔐 AnswerAI Login")

    menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])
    users = load_users()

    if menu == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                load_user_chat(username)
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    else:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Sign Up"):
            if new_user in users:
                st.warning("Username already exists")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("✅ Account created. Please login.")

# -----------------------------
# LOAD USER CHAT
# -----------------------------
def load_user_chat(username):
    chat_db = load_chat_history()
    if username not in chat_db:
        chat_db[username] = []
        save_chat_history(chat_db)
    st.session_state.messages = chat_db[username]

# -----------------------------
# SAVE USER CHAT
# -----------------------------
def save_user_chat():
    chat_db = load_chat_history()
    chat_db[st.session_state.username] = st.session_state.messages
    save_chat_history(chat_db)

# -----------------------------
# OLLAMA
# -----------------------------
def ask_mistral(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        data = response.json()
        return data.get("response", "Error in response")
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"

# -----------------------------
# KEYWORD SCORE
# -----------------------------
def keyword_score(query, text):
    return len(set(query.lower().split()) & set(text.lower().split()))

# -----------------------------
# MAIN APP
# -----------------------------
def main_app():

    st.title("🚀 AnswerAI")

    # Sidebar
    st.sidebar.write(f"👤 {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.messages = []
        save_user_chat()

    # -----------------------------
    # FILE UPLOAD
    # -----------------------------
    st.sidebar.header("📂 Upload Documents")

    files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if files:

        all_chunks = []

        for file in files:
            pages = extract_text_with_pages(file)

            if not pages:
                st.sidebar.warning(f"No text in {file.name}")
                continue

            for p in pages:
                if not p["text"].strip():
                    continue

                chunks = chunk_text(p["text"], chunk_size=700)

                for c in chunks:
                    if c.strip():
                        all_chunks.append({
                            "text": c,
                            "source": file.name,
                            "page": p["page"]
                        })

        if len(all_chunks) == 0:
            st.sidebar.error("No valid text found")
            return

        texts = [c["text"] for c in all_chunks]

        with st.spinner("Generating embeddings..."):
            embeddings = generate_embeddings(texts)

        if embeddings is None or len(embeddings) == 0:
            st.sidebar.error("❌ Embedding failed")
            return

        vs = VectorStore(len(embeddings[0]))
        vs.add_embeddings(embeddings, texts)

        st.session_state.vs = vs
        st.session_state.chunks = all_chunks

        st.sidebar.success(f"Indexed {len(all_chunks)} chunks")

    # -----------------------------
    # DISPLAY CHAT
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # -----------------------------
    # INPUT
    # -----------------------------
    query = st.chat_input("Ask something...")


    if query:

        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        if "vs" not in st.session_state:
            answer = "⚠️ Upload documents first"

        else:

            with st.spinner("Thinking..."):

                # 🔥 Include chat memory
                history_text = "\n".join([
                    f"{m['role']}: {m['content']}"
                    for m in st.session_state.messages[-6:]
                ])

                q_emb = model.encode(query)
                results = st.session_state.vs.search(q_emb, top_k=15)

                if not results:
                    answer = "No relevant info found"
                else:

                    scored = []

                    for r in results:
                        text = r["text"]

                        doc_emb = model.encode(text)

                        sem_score = float(
                            np.dot(q_emb, doc_emb) /
                            (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb))
                        )

                        key_score = keyword_score(query, text)
                        final_score = sem_score + (0.3 * key_score)

                        scored.append((final_score, text))

                    scored.sort(key=lambda x: x[0], reverse=True)

                    top_chunks = scored[:5]

                    selected_chunks = []
                    sources = set()

                    for _, text in top_chunks:
                        for c in st.session_state.chunks:
                            if c["text"] == text:
                                selected_chunks.append(c)
                                sources.add((c["source"], c["page"]))
                                break

                    context = "\n\n".join([
                        f"[{c['source']} - Page {c['page']}]\n{c['text']}"
                        for c in selected_chunks
                    ])

                    prompt = f"""
                        You are a highly accurate document-based AI assistant.

                        IMPORTANT RULES:
                        1. Answer ONLY using the provided document context
                        2. DO NOT use outside knowledge
                        3. DO NOT guess or assume anything
                        4. If the answer is partially available, explain as much as possible from the context
                        5. If answer is not present, say:
                        "Answer not found in the provided document"
                        6. Don,t use your knowledge, only use the context provided below
                        7. Strictly follow the ANSWER STYLE below
                        8.don,t answer if not directly found in the context, say "Not enough relevant information found in the document."

                        ANSWER STYLE:
                        - say "Answer not found in the provided document" and do not provide sources, if answer is not directly found
                        - Give detailed explanation
                        - Use clear paragraphs
                        - Include all relevant points from context
                        - Do not add extra information beyond the context
                        

                        ---------------------
                        CONVERSATION HISTORY:
                        {history_text}
                        ---------------------

                        DOCUMENT CONTEXT:
                        {context}
                        ---------------------

                        QUESTION:
                        {query}

                        DETAILED ANSWER:
                        - Explanation:
                        - Headings and Subheadings:
                        - Key Points:
                        - Conclusion:
"""

                    if len(context.strip()) < 50:
                        answer = "Not enough relevant information found in the document."
                    else:
                        answer = ask_mistral(prompt)
                    
                    if len(context.strip()) < 50:
                        answer = "Not enough relevant information found in the document."
                    else:
                        answer = ask_mistral(prompt)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # ✅ Save chat per user
        save_user_chat()

        if "vs" in st.session_state:
            with st.expander("📄 Sources"):
                for s in sources:
                    st.write(s)

# -----------------------------
# APP CONTROL
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_signup()