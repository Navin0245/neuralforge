# Training Infrastructure Notes

**Author:** Navin C Chacko
**Context:** Trainer class design and MLflow integration

---

## 1. The Trainer Class — What It Does

A Trainer encapsulates the entire training process.
It takes all individual components and orchestrates them.

**Responsibilities:**
1. Setup — move model to device, initialise trackers
2. Training phase per epoch — model.train(), 5-step loop
3. Validation phase per epoch — model.eval(), no_grad()
4. LR scheduling — scheduler.step() after each epoch
5. Logging — MLflow metrics per epoch
6. Checkpointing — save best_model.pth on val improvement

---

## 2. The 5-Step Optimization Loop

```python
for x, y_true in train_loader:
    x, y_true = x.to(device), y_true.to(device)

    # Step 1: Zero gradients (clear previous step)
    optimizer.zero_grad()

    # Step 2: Forward pass
    y_pred = model(x)

    # Step 3: Compute loss
    loss = loss_fn(y_pred, y_true)

    # Step 4: Backward pass (compute gradients)
    loss.backward()

    # Step 5: Update weights
    optimizer.step()
```

**Why zero_grad() first:**
PyTorch accumulates gradients by default.
Without zeroing, gradients from step t add to step t-1's gradients.
This causes incorrect weight updates — always zero first.

---

## 3. Epoch vs Step

| Concept | Definition | Example (N=1000, batch=20) |
|---------|-----------|--------------------------|
| Step | One weight update = one batch | 1000/20 = 50 steps |
| Epoch | One full pass over training data | 50 steps = 1 epoch |
| Training run | All epochs | 500 epochs = 25,000 steps |

**Outer loop = epochs. Inner loop = steps (batches).**

---

## 4. Weighted Batch Averaging — The Correct Way

The last batch in a DataLoader may be smaller than batch_size.
Simple averaging (sum/num_batches) gives wrong results.

**Wrong:**
```python
total_loss += loss.item()
avg_loss = total_loss / len(dataloader)   # divides by num_batches
# WRONG if last batch has fewer samples
```

**Correct:**
```python
# Weighted average: weight each batch by its actual size
total_loss    += loss.item() * current_batch_size
total_samples += current_batch_size
avg_loss = total_loss / total_samples   # divides by num_samples
```

This gives the true per-sample average regardless of batch sizes.

---

## 5. model.train() vs model.eval()

| Mode | Layers Affected | When To Use |
|------|----------------|-------------|
| model.train() | Dropout: active, BatchNorm: track stats | Training phase |
| model.eval() | Dropout: off, BatchNorm: use tracked stats | Validation/inference |

**Always set mode explicitly:**
```python
def _train_one_epoch(self):
    self.model.train()   # ← must set before loop
    ...

def _validate_one_epoch(self):
    self.model.eval()    # ← must set before loop
    with torch.no_grad():
        ...
```

---

## 6. torch.no_grad() — Why It Matters

```python
with torch.no_grad():
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
```

Without this context manager, PyTorch stores the entire computation
graph for every forward pass (for backpropagation).

During validation:
- You never call loss.backward()
- Storing the graph wastes memory (2-3× more)
- GPU memory that could hold larger batches is wasted

**Rule:** All inference and validation code goes inside torch.no_grad().

---

## 7. MLflow — What It Logs and Why

**The "spreadsheet problem":**
Without MLflow, you manually write results in Excel.
If a run crashes at epoch 300, those results are lost.
Comparing 20 hyperparameter runs requires 20 Excel rows.

**What MLflow records automatically:**
```
Parameters (logged once at start):
  da, du, d_v, k_max, n_layers, lr, epochs, device...

Metrics (logged every epoch):
  train_loss, val_loss, learning_rate

Artifacts (saved on best val):
  best_model.pth
```

**Viewing results:**
```
mlflow ui
# Opens browser at http://localhost:5000
# Shows all runs, compare loss curves, sort by best val loss
```

---

## 8. Checkpointing — What Gets Saved

```python
torch.save(model.state_dict(), "best_model.pth")
```

`state_dict()` saves only the learnable parameters — not the model code.
This is intentional: the code may change, but the weights are permanent.

**Loading later:**
```python
model = FNO1D(da=2, du=1, d_v=64, k_max=16)   # recreate architecture
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

**Always save only ONE file (best_model.pth):**
Saving model_epoch_1.pth, model_epoch_2.pth, etc. creates hundreds of
500MB files. Overwrite the same file — you only need the best checkpoint.

---

## 9. The StepLR Schedule

Paper uses StepLR: halve learning rate every 100 epochs.

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,   # every 100 epochs
    gamma=0.5,       # multiply by 0.5 (halve)
)
```

**LR at each epoch:**
```
Epoch 1-100:   lr = 0.001
Epoch 101-200: lr = 0.0005
Epoch 201-300: lr = 0.00025
Epoch 301-400: lr = 0.000125
Epoch 401-500: lr = 0.0000625
```

**Why decrease LR:**
Early training: large LR → fast rough convergence
Late training: small LR → fine-grained convergence to minimum
Constant LR: oscillates around minimum without settling

---

## 10. Config-Driven Design

**Problem with hardcoded values:**
```python
model = FNO1D(da=2, du=1, d_v=64, k_max=16)  # buried in script
# For ablation: must edit source code, easy to lose track
```

**Solution — YAML config:**
```yaml
# configs/burgers_fno.yaml
model:
  da: 2
  d_v: 64
  k_max: 16

training:
  epochs: 500
  lr: 0.001
```

```python
# experiments/burgers_fno1d.py
cfg = yaml.safe_load(open("configs/burgers_fno.yaml"))
model = FNO1D(**cfg["model"])
```

**Ablation runs:**
```bash
# Original
python experiments/burgers_fno1d.py --config configs/burgers_fno.yaml

# Test k_max=8
python experiments/burgers_fno1d.py --config configs/burgers_kmax8.yaml

# Test dv=32
python experiments/burgers_fno1d.py --config configs/burgers_dv32.yaml
```

MLflow records which config produced which result. Complete reproducibility.
