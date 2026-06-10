import os
import json
import time
import tempfile
import numpy as np

from mini_torch import xp, as_backend_array, asnumpy, is_gpu_available
from mini_torch import DataLoader
from TextCorpusDataset import TextCorpusDataset
from GPTModel import GPTModel
from CrossEntropyLoss import CrossEntropyLoss
from RMSprop import RMSprop

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # block_size sweep (num_heads=4 fixed)
    {"block_size": 16,  "num_heads": 4},
    {"block_size": 32,  "num_heads": 4},   # baseline
    {"block_size": 64,  "num_heads": 4},
    {"block_size": 128, "num_heads": 4},
    {"block_size": 256, "num_heads": 4},
    # num_heads sweep (block_size=32 fixed)
    {"block_size": 32,  "num_heads": 2},
    {"block_size": 32,  "num_heads": 8},
]

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
CORPUS_PATH   = "../Transformers/Pride-and-Prejudice-Cleaned.txt"
RESULTS_PATH  = "experiments/results.json"
WEIGHTS_DIR   = "experiments/weights"

EPOCHS          = 20
LEARNING_RATE   = 0.001
BATCH_SIZE      = 128 if is_gpu_available else 16
EMB_DIM         = 256 if is_gpu_available else 64
NUM_LAYERS      = 4   if is_gpu_available else 2
TRAIN_VAL_SPLIT = 0.9

NUM_FINAL_SAMPLES  = 5    # stochastic samples generated after training
FINAL_SAMPLE_LEN   = 400  # characters per final sample
TRAIN_SAMPLE_LEN   = 100  # characters shown per epoch during training
FINAL_TEMPERATURES = [0.0, 0.8]  # 0.0 = greedy (1 sample), 0.8 = stochastic


def run_name(cfg):
    return f"block{cfg['block_size']}_heads{cfg['num_heads']}"


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------
def load_corpus_split(block_size):
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    split = int(len(text) * TRAIN_VAL_SPLIT)
    train_text, val_text = text[:split], text[split:]

    # Train dataset via TextCorpusDataset (builds the authoritative tokenizer)
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    tf.write(train_text); tf.close()
    train_ds = TextCorpusDataset(tf.name, block_size)
    os.unlink(tf.name)

    # Val pairs encoded with the train tokenizer (consistent char→index mapping)
    stoi = train_ds.tokenizer.stoi
    val_tokens = np.array(
        [stoi[c] for c in val_text if c in stoi],
        dtype=np.int32
    )
    val_pairs = [
        (val_tokens[i : i + block_size], val_tokens[i + 1 : i + block_size + 1])
        for i in range(0, len(val_tokens) - block_size, block_size)
    ]

    return train_ds, val_pairs


# ---------------------------------------------------------------------------
# Validation loss (forward-only, no weight updates)
# ---------------------------------------------------------------------------
def compute_val_loss(model, val_pairs, loss_fn, vocab_size):
    total_loss = 0.0
    n_batches  = 0
    for i in range(0, len(val_pairs) - BATCH_SIZE + 1, BATCH_SIZE):
        batch       = val_pairs[i : i + BATCH_SIZE]
        x           = as_backend_array(np.stack([p[0] for p in batch]))
        y           = as_backend_array(np.stack([p[1] for p in batch]))
        logits      = model.forward(x)
        logits_flat = logits.reshape(-1, vocab_size)
        y_flat      = y.reshape(-1)
        y_onehot    = xp.eye(vocab_size, dtype=xp.float32)[y_flat]
        loss        = loss_fn.forward(logits_flat, y_onehot)
        total_loss += float(asnumpy(loss))
        n_batches  += 1
    return total_loss / n_batches if n_batches > 0 else float('inf')


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------
def generate_sample(model, tokenizer, length, temperature=0.8):
    start_idx     = tokenizer.encode('\n')
    start_context = xp.array([start_idx])
    out = model.generate(start_context, max_new_tokens=length, temperature=temperature)
    return tokenizer.decode(asnumpy(out)[0].tolist())


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_one_run(cfg, train_ds, val_pairs, vocab_size):
    model       = GPTModel(vocab_size, cfg["block_size"], EMB_DIM, cfg["num_heads"], NUM_LAYERS)
    loss_fn     = CrossEntropyLoss()
    val_loss_fn = CrossEntropyLoss()
    optimizer   = RMSprop([model], lr=LEARNING_RATE)
    train_dl    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    n_train     = len(train_dl)

    metrics = {
        "epoch_train_losses": [],
        "epoch_val_losses":   [],
        "epoch_val_ppls":     [],
        "epoch_times":        [],
        "epoch_samples":      [],
    }

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        total_train_loss = 0.0

        for x, y in train_dl:
            x = as_backend_array(x)
            y = as_backend_array(y)

            logits      = model.forward(x)
            logits_flat = logits.reshape(-1, vocab_size)
            y_flat      = y.reshape(-1)
            y_one_hot   = xp.eye(vocab_size, dtype=xp.float32)[y_flat]

            loss = loss_fn.forward(logits_flat, y_one_hot)
            total_train_loss += float(asnumpy(loss))

            grad_flat = loss_fn.backward()
            grad      = grad_flat.reshape(x.shape[0], x.shape[1], vocab_size)

            optimizer.zero_grad()
            model.backward(grad)
            optimizer.step()

        train_loss = total_train_loss / n_train
        val_loss   = compute_val_loss(model, val_pairs, val_loss_fn, vocab_size)
        val_ppl    = float(np.exp(val_loss))
        elapsed    = time.perf_counter() - t0
        sample     = generate_sample(model, train_ds.tokenizer, TRAIN_SAMPLE_LEN)

        metrics["epoch_train_losses"].append(round(train_loss, 6))
        metrics["epoch_val_losses"].append(round(val_loss, 6))
        metrics["epoch_val_ppls"].append(round(val_ppl, 4))
        metrics["epoch_times"].append(round(elapsed, 2))
        metrics["epoch_samples"].append(sample)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | "
              f"{elapsed:.1f}s")
        print(f"  Sample: {sample!r}")

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    for cfg in EXPERIMENTS:
        name = run_name(cfg)

        if name in results:
            print(f"[SKIP] {name} already in results.json\n")
            continue

        print(f"=== {name} | block_size={cfg['block_size']} num_heads={cfg['num_heads']} ===")

        train_ds, val_pairs = load_corpus_split(cfg["block_size"])
        vocab_size = train_ds.tokenizer.vocab_size
        print(f"  Vocab: {vocab_size} | Train seqs: {len(train_ds)} | Val pairs: {len(val_pairs)}")

        run_start = time.perf_counter()
        model, metrics = train_one_run(cfg, train_ds, val_pairs, vocab_size)
        total_time = time.perf_counter() - run_start

        weights_path = os.path.join(WEIGHTS_DIR, f"{name}.pkl")
        model.save_weights(weights_path)
        print(f"  Weights saved to {weights_path}")

        # Final samples: greedy (1 sample) and stochastic (NUM_FINAL_SAMPLES)
        print(f"  Generating final samples...")
        final_samples = {}
        for temp in FINAL_TEMPERATURES:
            n = 1 if temp <= 0 else NUM_FINAL_SAMPLES
            final_samples[str(temp)] = [
                generate_sample(model, train_ds.tokenizer, FINAL_SAMPLE_LEN, temp)
                for _ in range(n)
            ]

        results[name] = {
            "config": {
                "block_size":    cfg["block_size"],
                "num_heads":     cfg["num_heads"],
                "emb_dim":       EMB_DIM,
                "num_layers":    NUM_LAYERS,
                "batch_size":    BATCH_SIZE,
                "epochs":        EPOCHS,
                "learning_rate": LEARNING_RATE,
            },
            "total_time_s":       round(total_time, 1),
            "final_train_loss":   metrics["epoch_train_losses"][-1],
            "final_val_loss":     metrics["epoch_val_losses"][-1],
            "final_val_ppl":      metrics["epoch_val_ppls"][-1],
            **metrics,
            "final_samples":      final_samples,
            "weights_path":       weights_path,
        }

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved. Total run time: {total_time/60:.1f} min\n")

    print("All experiments complete.")


if __name__ == "__main__":
    main()
