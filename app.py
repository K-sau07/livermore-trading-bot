import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import torch, json, os, warnings
warnings.filterwarnings("ignore")
from transformer_model import LivermoreTransformer, SimpleTokenizer

st.set_page_config(
    page_title="Livermore Trading AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Loaders ──
@st.cache_data
def load_data():
    return pd.read_csv("Livermore_Data_Final.csv")

@st.cache_resource
def load_emb():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def build_emb(qs):
    return load_emb().encode(qs, show_progress_bar=True, batch_size=256)

@st.cache_resource
def build_tfidf(qs):
    v = TfidfVectorizer(stop_words="english", max_features=10000)
    return v, v.fit_transform(qs)

@st.cache_resource
def load_transformer():
    d = "model"
    fps = [f"{d}/config.json", f"{d}/transformer_weights.pt", f"{d}/tokenizer.json"]
    if not all(os.path.exists(f) for f in fps): return None, None
    with open(fps[0]) as f: cfg = json.load(f)
    tok = SimpleTokenizer(); tok.load(fps[2])
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    m = LivermoreTransformer(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"],
        cfg["num_layers"], cfg["d_ff"], cfg["max_len"], cfg["dropout"], cfg["pad_idx"]).to(dev)
    m.load_state_dict(torch.load(fps[1], map_location=dev)); m.eval()
    return m, tok

# ── Retrieval ──
def retrieve(q, embs, vec, tm, df, k=5, lf=None):
    qe = load_emb().encode([q])
    es = np.dot(embs, qe.T).flatten()
    es /= (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-10)
    ts = cosine_similarity(vec.transform([q]), tm).flatten()
    sc = 0.4 * ts + 0.6 * es
    if lf and lf != "All": sc *= (df["label"] == lf).values
    idx = sc.argsort()[-k:][::-1]
    return [{"question": df.iloc[i]["question"], "answer": df.iloc[i]["answer"],
             "label": df.iloc[i]["label"], "score": float(sc[i])} for i in idx if sc[i] > 0]

# ── Generators ──
def gen_rag(q, res):
    if not res: return "No relevant info found."
    r = res[0]["answer"]
    if res[0]["score"] <= 0.5 and len(res) > 1: r += f"\n\n{res[1]['answer']}"
    return r

def gen_trans(q):
    m, tok = load_transformer()
    if m is None: return "Transformer not trained. Run: python train_transformer.py"
    dev = next(m.parameters()).device
    with open("model/config.json") as f: cfg = json.load(f)
    ml = cfg.get("max_len", 96)
    src = tok.pad_sequence(tok.encode(q, ml), ml)
    with torch.no_grad():
        out = m.generate(torch.tensor([src], dtype=torch.long).to(dev), ml, 1, 2,
                         temperature=0.5, top_k=10)
    a = tok.decode(out[0])
    return a if len(a.strip()) >= 10 else "Transformer could not generate a confident response."

def gen_llm(q, res, key):
    if not res: return "No relevant info found."
    if not key: return "Add Groq API key in the sidebar to enable LLM."
    ctx = "\n\n".join([r["answer"] for r in res])
    try:
        c = Groq(api_key=key)
        r = c.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"You are Jesse Livermore's trading assistant. Answer using ONLY this info, be direct, 3-5 sentences:\n\n{ctx}\n\nQ: {q}\nA:"}],
            temperature=0.3, max_tokens=500)
        return r.choices[0].message.content
    except Exception as e: return f"Error: {e}"

# ── Backtest ──
@st.cache_data(ttl=3600)
def run_bt(sym, sd, ed, ss, sl, bw):
    stk = yf.download(sym, sd, ed)
    if stk.empty: return None, None, None
    df = stk.xs(sym, axis=1, level='Ticker') if isinstance(stk.columns, pd.MultiIndex) else stk.copy()
    df['SMA_S'] = df['Close'].rolling(ss).mean()
    df['SMA_L'] = df['Close'].rolling(sl).mean()
    df['HI'] = df['Close'].rolling(bw).max()
    df['LO'] = df['Close'].rolling(bw).min()
    df['Pos'] = 0
    df.loc[(df['Close'] > df['HI'].shift(1)) & (df['Close'] > df['SMA_S']) & (df['Close'] > df['SMA_L']), 'Pos'] = 1
    df.loc[df['Close'] < df['LO'].shift(1), 'Pos'] = -1
    df['Pos'] = df['Pos'].fillna(0)
    df['R_BH'] = df['Close'].pct_change()
    df['R_S'] = df['R_BH'] * df['Pos']
    df['C_BH'] = df['R_BH'].cumsum()
    df['C_S'] = df['R_S'].cumsum()
    sr = df['R_S'].cumsum().iloc[-1]
    bhr = df['R_BH'].cumsum().iloc[-1]
    s = {"sr": f"{sr:.2%}", "bhr": f"{bhr:.2%}",
         "sharpe": f"{df['R_S'].mean() / df['R_S'].std() * np.sqrt(252):.2f}" if df['R_S'].std() > 0 else "N/A",
         "days": len(df), "active": int((df['Pos'] != 0).sum())}
    return df, s, sym

# ══════════════════════════════════════════
#  SIDEBAR — Always visible
# ══════════════════════════════════════════
with st.sidebar:
    st.title("Livermore Trading AI")
    st.caption("GenAI-Powered Trading Intelligence Platform")
    st.divider()
    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get a free key at console.groq.com")
    st.divider()
    page = st.radio("Navigation",
                     ["Chatbot", "Backtest Dashboard"],
                     captions=["Ask Livermore anything", "Test trading strategies"],
                     label_visibility="visible")

# ══════════════════════════════════════════
#  CHATBOT PAGE
# ══════════════════════════════════════════
if page == "Chatbot":
    qa = load_data()
    with st.spinner("Loading AI models..."):
        embs = build_emb(qa["question"].tolist())
        vec, tm = build_tfidf(qa["question"].tolist())
    t_ok = os.path.exists("model/transformer_weights.pt")

    # -- Sidebar controls --
    with st.sidebar:
        st.divider()
        st.subheader("Retrieval Settings")
        lf = st.selectbox("Filter by Category",
                           ["All"] + sorted(qa["label"].unique().tolist()))
        tk = st.slider("Number of Sources", 1, 10, 5)

        st.divider()
        st.subheader("Model Status")
        st.write(":green[**RAG**] — Active")
        if t_ok:
            st.write(":green[**Transformer**] — Trained")
        else:
            st.write(":red[**Transformer**] — Not trained")
        if groq_key:
            st.write(":green[**LLM**] — Connected")
        else:
            st.write(":orange[**LLM**] — No API key")

        st.divider()
        st.subheader("Knowledge Base")
        st.metric("Total Q&A Pairs", f"{len(qa):,}")
        label_counts = qa["label"].value_counts()
        for label, count in label_counts.items():
            st.caption(f"{label}: {count:,}")

        st.divider()
        st.subheader("Sample Questions")
        for q in ["What was Livermore's first job?",
                   "How did Livermore manage risk?",
                   "What is a Pivotal Point?",
                   "How did Livermore handle losses?",
                   "What trading psychology did Livermore follow?"]:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state["sq"] = q

    # -- Main content --
    st.header("Livermore Trading Intelligence")
    st.markdown("Three AI models answer your questions **side by side** — compare RAG retrieval, a hand-coded Transformer, and LLM synthesis.")

    # Model info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.subheader("RAG", divider="blue")
            st.caption("Hybrid TF-IDF + Semantic Search. Retrieves the most relevant Q&A pair from the Livermore knowledge base.")
    with col2:
        with st.container(border=True):
            st.subheader("Transformer", divider="violet")
            st.caption("Hand-coded Encoder-Decoder Transformer trained from scratch on 19K+ Livermore Q&A pairs.")
    with col3:
        with st.container(border=True):
            st.subheader("RAG + LLM", divider="green")
            st.caption("Groq LLaMA 3.3 70B synthesizes RAG-retrieved context into a natural conversational answer.")

    st.divider()

    # -- Chat --
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    # Render chat history
    for msg in st.session_state.msgs:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown("**:blue[RAG Response]**")
                    st.write(msg["rag"])
                with r2:
                    st.markdown("**:violet[Transformer Response]**")
                    st.write(msg["trans"])
                with r3:
                    st.markdown("**:green[RAG + LLM Response]**")
                    st.write(msg["llm"])
                if msg.get("src"):
                    with st.expander(f"View {len(msg['src'])} retrieved sources"):
                        for s in msg["src"]:
                            st.markdown(f":gray[**{s['label']}**] — score: `{s['score']:.3f}`")
                            st.caption(s["question"])
            else:
                st.write(msg["content"])

    # Chat input
    dq = st.session_state.pop("sq", None)
    user_input = st.chat_input("Ask about Jesse Livermore's trading strategies, psychology, risk management...") or dq

    if user_input:
        st.session_state.msgs.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        res = retrieve(user_input, embs, vec, tm, qa, k=tk, lf=lf)

        with st.chat_message("assistant"):
            # Generate responses
            rag_r = gen_rag(user_input, res)
            with st.spinner("Transformer generating..."):
                trans_r = gen_trans(user_input)
            with st.spinner("LLM generating..."):
                llm_r = gen_llm(user_input, res, groq_key)

            # Display side by side
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("**:blue[RAG Response]**")
                st.write(rag_r)
            with r2:
                st.markdown("**:violet[Transformer Response]**")
                st.write(trans_r)
            with r3:
                st.markdown("**:green[RAG + LLM Response]**")
                st.write(llm_r)

            if res:
                with st.expander(f"View {len(res)} retrieved sources"):
                    for s in res:
                        st.markdown(f":gray[**{s['label']}**] — score: `{s['score']:.3f}`")
                        st.caption(s["question"])

        st.session_state.msgs.append({
            "role": "assistant", "rag": rag_r,
            "trans": trans_r, "llm": llm_r, "src": res
        })

# ══════════════════════════════════════════
#  BACKTEST PAGE
# ══════════════════════════════════════════
elif page == "Backtest Dashboard":
    st.header("Livermore Strategy Backtest")
    st.markdown("Back-test Jesse Livermore's **trend-following breakout strategy** on any stock.")

    with st.expander("How the Livermore Strategy Works"):
        st.markdown("""
**Entry Signal (BUY):** Close > Previous N-day High AND Close > Short MA AND Close > Long MA

**Exit Signal (SELL):** Close < Previous N-day Low

**Core Principle:** No averaging down — trade only in the direction of strength.
        """)

    with st.sidebar:
        st.divider()
        st.subheader("Backtest Settings")
        sym = st.text_input("Stock Ticker", value="AAPL").upper()

        st.caption("**Magnificent 7 — Quick Pick:**")
        mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        cs = st.columns(4)
        for i, t in enumerate(mag7):
            if cs[i % 4].button(t, key=f"bt_{t}"):
                st.session_state["_t"] = t
                st.rerun()
        if "_t" in st.session_state:
            sym = st.session_state.pop("_t")

        st.divider()
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start Date", value=date(2020, 1, 1))
        ed = c2.date_input("End Date", value=date(2025, 6, 30))
        st.divider()
        ss = st.number_input("Short MA Window", value=50, min_value=5, max_value=100)
        sl = st.number_input("Long MA Window", value=200, min_value=50, max_value=500)
        bw = st.number_input("Breakout Window (days)", value=20, min_value=5, max_value=100)

    run = st.button("Run Backtest", type="primary", use_container_width=True)

    if run or "btr" in st.session_state:
        if run:
            with st.spinner(f"Downloading {sym} data..."):
                bt, s, sy = run_bt(sym, datetime.combine(sd, datetime.min.time()),
                    datetime.combine(ed, datetime.min.time()), ss, sl, bw)
                if bt is not None:
                    st.session_state["btr"] = (bt, s, sy)
                else:
                    st.error(f"Could not find data for {sym}. Check the ticker.")
                    st.stop()

        bt, s, sy = st.session_state["btr"]

        # Metrics row
        st.subheader(f"Results — {sy}", divider="orange")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", s["sr"])
        m2.metric("Buy & Hold", s["bhr"])
        m3.metric("Sharpe Ratio", s["sharpe"])
        m4.metric("Active / Total Days", f"{s['active']} / {s['days']}")

        # Plotly layout
        pl = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12),
            margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified")

        # Chart 1 — Cumulative Returns
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['C_BH'], name='Buy & Hold',
            line=dict(color='#60a5fa', width=2)))
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['C_S'], name='Livermore Strategy',
            line=dict(color='#f97316', width=2), fill='tozeroy',
            fillcolor='rgba(249,115,22,0.08)'))
        fig1.update_layout(**pl, title=f"Cumulative Returns — {sy}",
            yaxis_title="Cumulative Return",
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — Price + MAs + Signals
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt.index, y=bt['Close'], name='Price',
            line=dict(color='#94a3b8', width=1)))
        fig2.add_trace(go.Scatter(x=bt.index, y=bt['SMA_S'], name=f'{ss}-day MA',
            line=dict(color='#f59e0b', width=1, dash='dash')))
        fig2.add_trace(go.Scatter(x=bt.index, y=bt['SMA_L'], name=f'{sl}-day MA',
            line=dict(color='#a78bfa', width=1, dash='dash')))
        buys = bt[bt['Pos'] == 1]
        sells = bt[bt['Pos'] == -1]
        fig2.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
            name='Buy Signal', marker=dict(color='#34d399', size=6, symbol='triangle-up')))
        fig2.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
            name='Sell Signal', marker=dict(color='#f87171', size=6, symbol='triangle-down')))
        fig2.update_layout(**pl, title=f"Price & Trade Signals — {sy}",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3 — Position timeline
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=bt.index, y=bt['Pos'], fill='tozeroy',
            fillcolor='rgba(52,211,153,0.2)', line=dict(color='#34d399', width=1)))
        fig3.update_layout(**pl, title="Position Over Time", height=220,
            yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Sell', 'Flat', 'Buy']))
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — Return distributions
        c1, c2 = st.columns(2)
        with c1:
            fig4a = go.Figure()
            fig4a.add_trace(go.Histogram(x=bt['R_BH'].dropna(), nbinsx=80,
                marker_color='#60a5fa', opacity=0.75))
            fig4a.update_layout(**pl, title="Buy & Hold — Daily Returns", height=300, showlegend=False)
            st.plotly_chart(fig4a, use_container_width=True)
        with c2:
            fig4b = go.Figure()
            fig4b.add_trace(go.Histogram(x=bt['R_S'].dropna(), nbinsx=80,
                marker_color='#f97316', opacity=0.75))
            fig4b.update_layout(**pl, title="Strategy — Daily Returns", height=300, showlegend=False)
            st.plotly_chart(fig4b, use_container_width=True)
