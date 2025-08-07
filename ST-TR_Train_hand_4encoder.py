# ==========================================================
# Action-Transformer (v2) – skeleton-keypoint classification
# ==========================================================
import os, random, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns

# -------------------------
# 1) Hyper-parameters
# -------------------------
DATA_PATH       = "Blue_Grutto_HandPose/"          # Match the output from KeypointDataProcess.py
#actions         = ['ASCEND','DESCEND','ME','STOP','ToRight','BUDDY_UP',
 #                  'FOLLOW_ME','OKAY','ToLeft','YOU','STAY']
actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'RIGHT', 'BUDDY_UP',
    'FOLLOW_ME', 'OKAY', 'LEFT', 'YOU', 'LEVEL'
]
SEQ_LEN         = 60               # Match the sequence_length from KeypointDataProcess.py
EMBED_DIM       = 128
N_HEADS         = 8
DEPTH           = 4
FF_DIM          = 256
DROPOUT_P       = 0.1
BATCH_SIZE      = 64
EPOCHS          = 100
MIXUP_ALPHA     = 0.4        # 0 ⇒ off
WARMUP_EPOCHS   = 5
BASE_LR         = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ----------------------------------------------------------
# 2) Load all sequences, compute per-joint μ/σ, normalise
# ----------------------------------------------------------
label_map  = {a:i for i,a in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    base_dir   = os.path.join(DATA_PATH, action)
    subfolders = sorted(int(f) for f in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, f)))
    for seq_id in subfolders:
        window = [np.load(os.path.join(base_dir, str(seq_id), f"{f}.npy"))
                  for f in range(SEQ_LEN)]
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)        # N × 60 × 126
y = np.array(labels,   dtype=np.int64)
N, feat_dim = X.shape[0], X.shape[-1]      # <- use a different name
print(f"Loaded {N} sequences; feature dim = {feat_dim}")

# ---- per-joint μ/σ along time & sample dims (axis=(0,1)) ----
mu  = X.mean(axis=(0,1), keepdims=True)
std = X.std (axis=(0,1), keepdims=True) + 1e-6
X   = (X - mu) / std                                    # N × 60 × 126

# -------------------------------------------
# 3) Train / test split  (stratified)
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

NUM_CLASSES = len(actions)

# ------------------------------------------------
# 4) Dataset class with on-the-fly augmentation
# ------------------------------------------------
def time_warp(seq, min_len=52, max_len=68):
    """Resample sequence to a random length then pad/truncate back."""
    L = random.randint(min_len, max_len)
    idx = np.linspace(0, SEQ_LEN-1, L).round().astype(int)
    warped = seq[idx]
    if L < SEQ_LEN:
        pad = np.repeat(warped[-1:], SEQ_LEN-L, axis=0)
        warped = np.concatenate([warped, pad], 0)
    return warped[:SEQ_LEN]

class GestureDataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        seq, label = self.X[i], self.y[i]
        if self.train:
            # ----- random 10 % frame drop OR time-warp -----
            if random.random() < 0.5:
                mask = np.sort(np.random.choice(SEQ_LEN,
                           int(SEQ_LEN*0.9), replace=False))
                seq = seq[mask]
                # pad to 30
                pad = np.repeat(seq[-1:], SEQ_LEN-len(seq), axis=0)
                seq = np.concatenate([seq, pad], 0)
            else:
                seq = time_warp(seq)

            scale = np.random.uniform(0.9, 1.1)
            seq *= scale
            seq += np.random.normal(0, 0.01, size=seq.shape)
        return torch.from_numpy(seq), label

train_ds = GestureDataset(X_train, y_train, train=True)
test_ds  = GestureDataset(X_test , y_test , train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds , batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------
# 5) Model – deeper pre-norm Transformer
# --------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pos = torch.arange(max_len)[:,None]
        i   = torch.arange(0,d_model,2)[None]
        div = torch.exp(i * (-math.log(10000.0)/d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))     # 1 × L × D
    def forward(self,x): return x + self.pe[:,:x.size(1)]

# ---------- model ----------
class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, EMBED_DIM)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))

        # ⬇️ use 60 (seq) + 1 (CLS) positions
        self.pos_enc = PositionalEncoding(EMBED_DIM, max_len=SEQ_LEN + 1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT_P, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, DEPTH)
        self.head    = nn.Sequential(nn.LayerNorm(EMBED_DIM),
                                     nn.Linear(EMBED_DIM, num_classes))

    def forward(self, x):                 # B × 60 × 126
        B = x.size(0)
        x = self.in_proj(x)
        cls = self.cls_tok.expand(B, -1, -1)
        x   = torch.cat([cls, x], 1)      # B × 61 × D
        x   = self.pos_enc(x)             # add 61-length PE
        x   = self.encoder(x)
        return self.head(x[:, 0])         # CLS
    
model = ActionTransformer(feat_dim, NUM_CLASSES).to(device)

# ----------------------
# 6) Optimiser & sched
# ----------------------
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch+1)/WARMUP_EPOCHS
    # cosine from 1 → 0
    progress = (epoch-WARMUP_EPOCHS)/(EPOCHS-WARMUP_EPOCHS)
    return 0.5*(1+math.cos(math.pi*progress))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------------
# 7) Helper functions
# ----------------------
def evaluate(net, loader):
    net.eval(); total, correct, loss_sum = 0,0,0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = net(Xb);  loss = criterion(out,yb)
            loss_sum += loss.item()*yb.size(0)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
            total   += yb.size(0)
    return loss_sum/total, correct/total

# ----------------------
# 8) Training loop
# ----------------------
train_hist, test_hist = [], []
for epoch in range(1, EPOCHS+1):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        # ----- MixUp -----
        if MIXUP_ALPHA > 0:
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(Xb.size(0)).to(device)
            X_mix = lam*Xb + (1-lam)*Xb[idx]
            y_one = F.one_hot(yb, NUM_CLASSES).float()
            y_mix = lam*y_one + (1-lam)*y_one[idx]
            logits = model(X_mix)
            loss = torch.sum(-y_mix*F.log_softmax(logits,1)) / Xb.size(0)
        else:
            logits = model(Xb)
            loss   = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # --- metrics ---
    tr_loss, tr_acc = evaluate(model, train_loader)
    te_loss, te_acc = evaluate(model, test_loader)
    train_hist.append((tr_loss,tr_acc)); test_hist.append((te_loss,te_acc))

    if epoch%10==0 or epoch==1:
        print(f"E{epoch:03d} | "
              f"Train {tr_loss:.3f}/{tr_acc*100:.1f}% | "
              f"Test {te_loss:.3f}/{te_acc*100:.1f}% | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

print("✔ training complete")

# -----------------------------
# 9) Curves & confusion matrix
# -----------------------------
tl, ta = zip(*train_hist); vl, va = zip(*test_hist)
plt.figure(figsize=(11,4))
plt.subplot(1,2,1); plt.plot(tl,label="train"); plt.plot(vl,label="test")
plt.title("Loss"); plt.legend(); plt.grid()
plt.subplot(1,2,2); plt.plot(ta,label="train"); plt.plot(va,label="test")
plt.title("Accuracy"); plt.legend(); plt.grid(); plt.show()

model.eval(); all_pred, all_true = [], []
with torch.no_grad():
    for Xb,yb in test_loader:
        p = model(Xb.to(device)).argmax(1).cpu()
        all_pred.append(p); all_true.append(yb)
y_pred = torch.cat(all_pred).numpy()
y_true = torch.cat(all_true).numpy()

# Confusion matrix (normalize row-wise, convert to %)
cm = confusion_matrix(y_true, y_pred, normalize="true") * 100

# Pre-format labels (not transposed)
cm_labels = np.array([["{:.1f} %".format(v) for v in row] for row in cm])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm.T,                  # transpose the actual heatmap values
    annot=cm_labels,       # but keep the original (non-transposed) labels
    fmt="",
    cmap="Blues",
    xticklabels=actions,   # Predicted  on x-axis
    yticklabels=actions    # True  on y-axis
)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

print(f"Final test accuracy: {accuracy_score(y_true,y_pred):.4f}")

# save weights
torch.save({"state_dict":model.state_dict(),
            "mu":mu, "std":std}, "action_transformer_v2.pth")
print("Model saved → action_transformer_v2.pth")