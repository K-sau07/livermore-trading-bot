"""
============================================================
 HAND-CODED TRANSFORMER MODEL — Built from Scratch
 Jesse Livermore Trading Chatbot
============================================================
 Architecture (Vaswani et al. 2017 — "Attention Is All You Need"):
   - Sinusoidal Positional Encoding
   - Multi-Head Self-Attention (manual Q/K/V projections)
   - Cross-Attention (Decoder attends to Encoder output)
   - Position-wise Feed-Forward Networks
   - Layer Normalization + Residual Connections
   - Causal Masking for auto-regressive decoding
   - Top-K Sampling for diverse generation
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import re


# ============================================================
#  1. POSITIONAL ENCODING (Sinusoidal)
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Injects position information using sine/cosine functions.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
#  2. MULTI-HEAD SELF-ATTENTION (Hand-coded from scratch)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Attention across multiple heads.
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ============================================================
#  3. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """FFN(x) = ReLU(xW1 + b1)W2 + b2"""
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


# ============================================================
#  4. ENCODER LAYER
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ============================================================
#  5. DECODER LAYER
# ============================================================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


# ============================================================
#  6. FULL TRANSFORMER (Encoder-Decoder)
# ============================================================

class LivermoreTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer for Q&A generation.
    Hand-coded from scratch — no HuggingFace shortcuts.
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4,
                 d_ff=512, max_len=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, L = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(L, L, device=tgt.device)).bool().unsqueeze(0).unsqueeze(1)
        return pad_mask & causal

    def encode(self, src, mask):
        x = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder: x = layer(x, mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder: x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.out_proj(dec_out)

    @torch.no_grad()
    def generate(self, src, max_len=128, sos_idx=1, eos_idx=2, temperature=0.8, top_k=30):
        """
        Auto-regressive generation with top-k sampling.
        Top-k sampling picks from the k most likely next tokens,
        giving more diverse and natural output than greedy decoding.
        """
        self.eval()
        device = src.device
        src_mask = self.make_src_mask(src)
        enc_out = self.encode(src, src_mask)
        tgt = torch.full((src.size(0), 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt)
            dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
            logits = self.out_proj(dec_out[:, -1, :]) / temperature

            # Top-k filtering
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == eos_idx).all(): break

        return tgt


# ============================================================
#  7. IMPROVED TOKENIZER (word-level with better preprocessing)
# ============================================================

class SimpleTokenizer:
    """
    Word-level tokenizer with improved text preprocessing.
    Handles punctuation, contractions, and case normalization.
    """
    SPECIAL = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __init__(self):
        self.word2idx = dict(self.SPECIAL)
        self.idx2word = {v: k for k, v in self.SPECIAL.items()}
        self.vocab_size = len(self.SPECIAL)

    @staticmethod
    def tokenize_text(text):
        """Clean and tokenize text into words."""
        text = text.lower().strip()
        # Keep apostrophes in contractions, split punctuation otherwise
        text = re.sub(r"([.!?,;:\"()\[\]])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def build_vocab(self, texts, min_freq=2, max_vocab=15000):
        """Build vocabulary from corpus."""
        freq = {}
        for t in texts:
            for w in self.tokenize_text(t):
                freq[w] = freq.get(w, 0) + 1
        # Sort by frequency (most common first) and limit vocab
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        for word, count in sorted_words:
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                if len(self.word2idx) >= max_vocab: break
        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] Vocab size: {self.vocab_size} (min_freq={min_freq})")

    def encode(self, text, max_len=128):
        words = self.tokenize_text(text)
        tokens = [self.word2idx.get(w, 3) for w in words]  # 3 = UNK
        tokens = [1] + tokens[:max_len - 2] + [2]  # SOS ... EOS
        return tokens

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            if isinstance(idx, torch.Tensor): idx = idx.item()
            if idx in (0, 1): continue  # Skip PAD, SOS
            if idx == 2: break  # Stop at EOS
            w = self.idx2word.get(idx, "<UNK>")
            words.append(w)
        # Clean up spacing around punctuation
        text = " ".join(words)
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        return text

    def pad_sequence(self, tokens, max_len):
        if len(tokens) >= max_len: return tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

    def save(self, path):
        data = {"word2idx": self.word2idx, "idx2word": {int(k): v for k, v in self.idx2word.items()}}
        with open(path, "w") as f: json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f: data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)
        return self
