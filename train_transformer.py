"""
============================================================
 FAST TRAINING — Optimized for 30 minutes
 Memorization-focused: trains the Transformer to closely
 reproduce answers from the knowledge base.
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os, time, json
from transformer_model import LivermoreTransformer, SimpleTokenizer


class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_len=96):
        self.tok = tokenizer
        self.ml = max_len
        self.pairs = list(zip(questions, answers))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        src = self.tok.pad_sequence(self.tok.encode(q, self.ml), self.ml)
        tgt = self.tok.pad_sequence(self.tok.encode(a, self.ml), self.ml)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def train():
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    MAX_LEN = 80
    D_MODEL = 192
    NUM_HEADS = 6
    NUM_LAYERS = 3
    D_FF = 384
    DROPOUT = 0.05  # Low dropout = more memorization
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 55)
    print("  LIVERMORE TRANSFORMER — Fast Training (30 min)")
    print("=" * 55)
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  Model: d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} dropout={DROPOUT}")
    print("=" * 55)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv("Livermore_Data_Final.csv")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    print(f"    {len(questions):,} Q&A pairs")

    # Tokenizer
    print("[2] Building vocab...")
    tok = SimpleTokenizer()
    tok.build_vocab(questions + answers, min_freq=2, max_vocab=12000)

    # Dataset
    print("[3] Preparing data...")
    ds = QADataset(questions, answers, tok, MAX_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    print(f"    {len(dl)} batches/epoch")

    # Model
    print("[4] Building model...")
    model = LivermoreTransformer(
        vocab_size=tok.vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_FF, max_len=MAX_LEN,
        dropout=DROPOUT, pad_idx=0
    ).to(DEVICE)
    print(f"    {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer — plain CE loss (no label smoothing = sharper memorization)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Train
    print("[5] Training...\n")
    os.makedirs("model", exist_ok=True)
    best = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0
        t0 = time.time()

        for src, tgt in dl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(dl)
        scheduler.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        pct = int(30 * ep / EPOCHS)
        print(f"  [{'#'*pct}{'-'*(30-pct)}] {ep:2d}/{EPOCHS} | Loss: {avg:.4f} | LR: {lr:.6f} | {dt:.0f}s")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), "model/transformer_weights.pt")
            tok.save("model/tokenizer.json")
            json.dump({"vocab_size": tok.vocab_size, "d_model": D_MODEL,
                        "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
                        "d_ff": D_FF, "max_len": MAX_LEN,
                        "dropout": DROPOUT, "pad_idx": 0},
                       open("model/config.json", "w"), indent=2)
            print(f"  ** Saved (loss={best:.4f})")

    print(f"\n{'='*55}")
    print(f"  Done! Best loss: {best:.4f}")
    print(f"{'='*55}")

    # Quick test
    print("\n  Sample outputs:\n")
    model.eval()
    for q in ["What was Livermore's first job?",
              "How did Livermore manage risk?",
              "What is a Pivotal Point?",
              "How did Livermore handle losses?"]:
        src = tok.pad_sequence(tok.encode(q, MAX_LEN), MAX_LEN)
        t = torch.tensor([src], dtype=torch.long).to(DEVICE)
        out = model.generate(t, max_len=MAX_LEN, sos_idx=1, eos_idx=2, temperature=0.5, top_k=10)
        print(f"  Q: {q}")
        print(f"  A: {tok.decode(out[0])}\n")


if __name__ == "__main__":
    train()
