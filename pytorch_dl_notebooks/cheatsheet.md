# PyTorch Notes

## 1. Core Philosophy

* **Define-by-run (dynamic graph)**: Graph is built during execution.
* Easier debugging than static graphs.
* Python-first, research-friendly.

---

## 2. Tensors

* `torch.tensor(data, dtype, device)`
* Similar to NumPy but supports GPU and autograd.
* Common ops:

  * Shape: `x.shape`, `x.view()`, `x.reshape()`
  * Device: `x.to(device)`
  * NumPy bridge: `x.numpy()`, `torch.from_numpy()`

---

## 3. Autograd

* Automatic differentiation engine.
* `requires_grad=True` tracks gradients.
* Backprop:

  ```python
  loss.backward()
  ```
* Grad stored in `tensor.grad`
* Stop tracking:

  * `with torch.no_grad():`
  * `tensor.detach()`

---

## 4. Neural Network Module (`nn.Module`)

* Base class for all models.
* Key components:

  * `__init__()` → define layers
  * `forward()` → define computation
* Example:

  ```python
  class Net(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc = nn.Linear(10, 1)
      def forward(self, x):
          return self.fc(x)
  ```

---

## 5. Layers (`torch.nn`)

* Common layers:

  * `nn.Linear`
  * `nn.Conv2d`
  * `nn.RNN`, `nn.LSTM`, `nn.GRU`
  * `nn.Embedding`
* Activations:

  * `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax(dim)`

---

## 6. Loss Functions

* Regression:

  * `nn.MSELoss`, `nn.L1Loss`
* Classification:

  * `nn.CrossEntropyLoss` (includes Softmax)
  * `nn.BCELoss`, `nn.BCEWithLogitsLoss`

---

## 7. Optimizers

* Defined after model parameters:

  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  ```
* Common:

  * `SGD`, `Adam`, `RMSprop`
* Update step:

  ```python
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```

---

## 8. Training Loop (Canonical)

```python
for epoch in range(epochs):
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
```

---

## 9. Dataset & DataLoader

* Custom dataset:

  ```python
  class MyDataset(Dataset):
      def __len__(self): return len(self.X)
      def __getitem__(self, idx): return self.X[idx], self.y[idx]
  ```
* `DataLoader(dataset, batch_size, shuffle)`

---

## 10. Model Evaluation

* Switch modes:

  * `model.train()`
  * `model.eval()`
* Disable gradients during eval:

  ```python
  with torch.no_grad():
      outputs = model(X)
  ```

---

## 11. Saving & Loading Models

* Save weights:

  ```python
  torch.save(model.state_dict(), 'model.pt')
  ```
* Load:

  ```python
  model.load_state_dict(torch.load('model.pt'))
  ```

---

## 12. Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

* Apple Silicon: `mps`

---

## 13. Common Pitfalls

* Forgetting `optimizer.zero_grad()`
* Shape mismatch in loss
* Using `CrossEntropyLoss` with one-hot targets (wrong)
* Forgetting `model.eval()` during inference

---

## 14. TensorFlow → PyTorch Mapping

| TensorFlow     | PyTorch              |
| -------------- | -------------------- |
| `Model.fit()`  | Manual training loop |
| `GradientTape` | Autograd             |
| `tf.data`      | Dataset + DataLoader |
| `Keras layers` | `nn.Module`          |

---

## 15. Mental Recall Checklist

* Define model → loss → optimizer
* Training loop = forward → loss → backward → step
* `train()` vs `eval()`
* Shapes always matter
* Loss decides output format

-----------
# -----------------------
# Pipeline


### Step 1: Define Problem

* Input shape
* Output shape
* Task type: regression / binary / multiclass

### Step 2: Prepare Data

* Convert to tensors
* Correct dtype (`float32`, `long`)
* Train / val split
* Batch using `DataLoader`

### Step 3: Build Model (`nn.Module`)

* Layers in `__init__`
* Computation in `forward()`

### Step 4: Define Loss

* Regression → `MSELoss`
* Binary → `BCEWithLogitsLoss`
* Multiclass → `CrossEntropyLoss`

### Step 5: Define Optimizer

* `Adam` (default choice)

### Step 6: Training Loop

* forward → loss → backward → step

### Step 7: Evaluation

* `model.eval()` + `torch.no_grad()`

---
----

# Minimal NN (Linear Model)



```python
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.fc(x)
```

---

##  ANN / MLP Pipeline

### Structure

* Input → Linear → Activation → Linear → Output

```python
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_out)
        )
    def forward(self, x):
        return self.net(x)
```

### Recall Rules

* No activation after last layer (if using CE or BCEWithLogits)
* Flatten input manually

---

##  CNN Pipeline

### Input Shape

* `(batch, channels, height, width)`

### Structure

* Conv → ReLU → Pool → Conv → Flatten → FC

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64*5*5, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Recall Rules

* Always calculate feature map size
* Use `.view(batch, -1)`

---

##  RNN / LSTM Pipeline

### Input Shape

* `(batch, seq_len, features)`

```python
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h, c) = self.rnn(x)
        return self.fc(out[:, -1, :])
```

### Recall Rules

* Use last timestep output
* `batch_first=True` avoids confusion

---

##  Training Loop (One Loop for ALL Models)

```python
for epoch in range(E):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
```

---

## G. Evaluation Template

```python
model.eval()
with torch.no_grad():
    preds = model(X)
```

---
