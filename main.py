"""VoiceScribe - Speech to Text with Streamlit + Hugging Face Whisper."""
import sqlite3
import time
from datetime import datetime

import librosa
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline

DB_PATH = "history.db"

MODELS = {
    "Whisper Large v3 Turbo (best balance, GPU recommended)": "openai/whisper-large-v3-turbo",
    "Whisper Large v3 (highest accuracy, slow on CPU)": "openai/whisper-large-v3",
    "Whisper Small (good accuracy, CPU friendly)": "openai/whisper-small",
    "Whisper Base (fastest, lower accuracy)": "openai/whisper-base",
}

LANGUAGES = {
    "Auto-detect": None, "English": "english", "Urdu": "urdu", "Hindi": "hindi",
    "Arabic": "arabic", "Spanish": "spanish", "French": "french",
    "German": "german", "Chinese": "chinese", "Turkish": "turkish",
}

st.set_page_config(page_title="VoiceScribe", page_icon="🎙️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {
    padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    color: white; box-shadow: 0 10px 30px rgba(99,102,241,.35);
}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: white;}
.hero p {margin: .4rem 0 0; opacity: .9; font-size: 1.05rem;}
.card {
    padding: 1.2rem 1.4rem; border-radius: 16px; margin: .6rem 0;
    background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);
}
.result {
    padding: 1.4rem; border-radius: 16px; font-size: 1.1rem; line-height: 1.7;
    background: rgba(99,102,241,.08); border-left: 5px solid #8b5cf6;
}
.meta {font-size: .8rem; opacity: .7;}
.badge {
    display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
    background: rgba(139,92,246,.18); color: #8b5cf6; margin-right: 6px;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px; font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Database ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, source TEXT, model TEXT, language TEXT,
            task TEXT, duration REAL, proc_time REAL, text TEXT)"""
    )
    return conn


def save_item(**kw):
    with db() as c:
        c.execute(
            "INSERT INTO history (created_at, source, model, language, task, duration, proc_time, text)"
            " VALUES (:created_at,:source,:model,:language,:task,:duration,:proc_time,:text)",
            kw,
        )


def load_history(query=""):
    with db() as c:
        return pd.read_sql_query(
            "SELECT * FROM history WHERE text LIKE ? ORDER BY id DESC", c, params=(f"%{query}%",)
        )


def delete_item(item_id):
    with db() as c:
        c.execute("DELETE FROM history WHERE id=?", (item_id,))


def clear_all():
    with db() as c:
        c.execute("DELETE FROM history")


# ---------- Model ----------
@st.cache_resource(show_spinner="Loading model (first time downloads it)...")
def load_model(model_id: str):
    use_gpu = torch.cuda.is_available()
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device="cuda:0" if use_gpu else "cpu",
    )


def transcribe(file_like, model_id, language, task):
    audio, sr = librosa.load(file_like, sr=16000, mono=True)
    pipe = load_model(model_id)
    gen = {"task": task}
    if language:
        gen["language"] = language
    start = time.time()
    out = pipe(
        {"raw": audio, "sampling_rate": sr},
        chunk_length_s=30, batch_size=8, generate_kwargs=gen,
    )
    return out["text"].strip(), len(audio) / sr, time.time() - start


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    model_label = st.selectbox("Model", list(MODELS))
    lang_label = st.selectbox("Spoken language", list(LANGUAGES))
    task_label = st.radio("Mode", ["Transcribe", "Translate to English"])
    st.caption(f"Running on: **{'GPU 🚀' if torch.cuda.is_available() else 'CPU 🐢'}**")
    st.info("Tip: no GPU? Use Whisper Small for the best speed/accuracy balance.")

model_id = MODELS[model_label]
language = LANGUAGES[lang_label]
task = "translate" if task_label.startswith("Translate") else "transcribe"

# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>🎙️ VoiceScribe</h1>'
    "<p>Record or upload audio and get accurate text powered by OpenAI Whisper.</p></div>",
    unsafe_allow_html=True,
)

tab_main, tab_hist = st.tabs(["🎤 Transcribe", "🕘 History"])

# ---------- Transcribe tab ----------
with tab_main:
    col1, col2 = st.columns(2, gap="large")
    audio_file, source = None, None

    with col1:
        st.subheader("Record")
        rec = st.audio_input("Click the mic and speak")
        if rec:
            audio_file, source = rec, "Microphone"

    with col2:
        st.subheader("Upload")
        up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "webm"])
        if up:
            st.audio(up)
            audio_file, source = up, f"Upload: {up.name}"

    if st.button("✨ Transcribe", type="primary", use_container_width=True, disabled=audio_file is None):
        try:
            with st.spinner("Transcribing..."):
                audio_file.seek(0)
                text, dur, proc = transcribe(audio_file, model_id, language, task)
            if text:
                save_item(
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source=source,
                    model=model_id, language=lang_label, task=task,
                    duration=round(dur, 1), proc_time=round(proc, 1), text=text,
                )
                st.session_state["last"] = (text, dur, proc)
            else:
                st.warning("No speech detected.")
        except Exception as e:
            st.error(f"Error: {e}")

    if "last" in st.session_state:
        text, dur, proc = st.session_state["last"]
        st.markdown("### Result")
        st.markdown(f'<div class="result">{text}</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Audio length", f"{dur:.1f}s")
        m2.metric("Processing time", f"{proc:.1f}s")
        m3.metric("Words", len(text.split()))
        st.text_area("Copy text", text, height=120)
        st.download_button("⬇️ Download .txt", text, "transcript.txt")

# ---------- History tab ----------
with tab_hist:
    top1, top2 = st.columns([3, 1])
    q = top1.text_input("🔍 Search history", placeholder="Type a word...")
    df = load_history(q)

    if df.empty:
        st.info("No transcriptions yet.")
    else:
        top2.download_button(
            "⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"),
            "history.csv", "text/csv", use_container_width=True,
        )
        for _, r in df.iterrows():
            with st.container():
                st.markdown(
                    f'<div class="card"><span class="badge">{r.language}</span>'
                    f'<span class="badge">{r.task}</span>'
                    f'<span class="meta">{r.created_at} • {r.source} • {r.duration}s audio • '
                    f'{r.model.split("/")[-1]}</span><p style="margin-top:.7rem">{r.text}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button("🗑️ Delete", key=f"del_{r.id}"):
                    delete_item(int(r.id))
                    st.rerun()
        st.divider()
        if st.button("Clear all history"):
            clear_all()
            st.rerun()
