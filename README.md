# 📈 Livermore Trading Intelligence

A GenAI-powered chatbot and stock backtesting platform built around **Jesse Livermore's** trading philosophy.

This project combines **RAG (Retrieval-Augmented Generation)**, a **Hand-Coded Transformer** (built from scratch in PyTorch), and **Groq LLaMA 3.3 70B** to answer questions about Livermore's life, strategies, and market wisdom.

---

## 🚀 Features

### 💬 Three-Model Chatbot (Side-by-Side Comparison)

| Model | Type | Description |
|-------|------|-------------|
| **🔍 RAG** | Retrieval | Hybrid TF-IDF + Semantic Search (`all-MiniLM-L6-v2`) retrieves best Q&A matches |
| **🧠 Transformer** | Generative | **Hand-coded Encoder-Decoder Transformer** trained on the Livermore dataset |
| **🤖 RAG + LLM** | Hybrid | Groq's LLaMA 3.3 70B synthesizes RAG results into natural answers |

### 🧠 Hand-Coded Transformer Architecture

Built **entirely from scratch** using PyTorch — no HuggingFace model imports:

- **Sinusoidal Positional Encoding** (Vaswani et al. 2017)
- **Multi-Head Self-Attention** with manual Q/K/V projections
- **Cross-Attention** between encoder and decoder
- **Feed-Forward Networks** with ReLU activation
- **Layer Normalization** + Residual Connections
- **Causal Masking** for auto-regressive decoding
- **Greedy Decoding** for inference

### 📊 Livermore Strategy Backtest
- Trend-following breakout strategy on any stock
- Magnificent 7 quick-pick buttons
- Configurable MA windows and breakout parameters
- 4 visualizations: Cumulative returns, Price+Signals, Position timeline, Return distributions

---

## 🛠️ Setup

### 1. Create Environment
```bash
cd Livermore-Trading-Bot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Transformer
```bash
python train_transformer.py
```
This trains the hand-coded Transformer on your Livermore Q&A data and saves weights to `./model/`.
Takes ~5-10 minutes on CPU, faster on GPU/MPS.

### 3. Get a Groq API Key (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up free → Generate API key
3. Enter it in the sidebar when running the app

> RAG and Transformer work without an API key. Only the RAG+LLM column needs Groq.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
Livermore-Trading-Bot/
├── app.py                    # Main Streamlit application (UI + all 3 models)
├── transformer_model.py      # Hand-coded Transformer architecture (from scratch)
├── train_transformer.py      # Training script for the Transformer
├── Livermore_Data_Final.csv   # Knowledge base (19K+ Q&A pairs)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── model/                    # Created after training
    ├── transformer_weights.pt
    ├── tokenizer.json
    └── config.json
```

---

## 🏗️ Architecture Diagram

```
User Question
     │
     ├──► TF-IDF Vectorizer ──► Keyword Scores ─┐
     │                                            ├──► Hybrid Ranking ──► Top-K Results
     └──► SentenceTransformer ──► Semantic Scores─┘
                                                          │
                    ┌─────────────────────────────────────┼──────────────────────┐
                    │                                     │                      │
                    ▼                                     ▼                      ▼
              🔍 RAG Response                    🧠 Transformer              🤖 RAG + LLM
            (Direct Retrieval)              (Hand-Coded Seq2Seq)       (Groq LLaMA 3.3 70B)
             Best matching QA              Encoder encodes question      LLM synthesizes RAG
             from knowledge base           Decoder generates answer      results naturally
```

---

## 📊 Dataset Categories

| Category | Description |
|----------|-------------|
| Personal Life | Biography, early career, family |
| Strategy Development | Trading systems, methods |
| Timing | Market timing, entry/exit signals |
| Psychology | Trading psychology, emotional discipline |
| Risk Management | Position sizing, loss cutting |
| Adaptability | Adapting to changing markets |

---

## 📝 License
For educational purposes only.
