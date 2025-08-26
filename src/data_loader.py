import pandas as pd
import numpy as np
import datetime
import random
import os

class DataLoader:
    def __init__(self, dataset_path, has_header: bool = True, expect_cols: int = 785, dtype_raw: str = "uint8", scale: str = "0to1", center: bool = False, data_split: dict = None, stratify: bool = True, seed: int = 42, orientation: str = "cols", cache_path: str | None = "data/processed/mnist.npz", version: str = "v1", num_features: int = 784, num_classes: int = 10, label_map: dict | None = None):
        self.dataset_path = dataset_path
        self.has_header = has_header
        
        if expect_cols == num_features + 1:
            self.expect_cols = expect_cols
        else:
            raise ValueError(f"Expected columns are bad: {expect_cols}")
        
        if dtype_raw == "uint8":
            self.dtype_raw = dtype_raw
        else:
            raise ValueError(f"dtype: {dtype_raw} not supported")
        
        if scale == '0to1' or scale == '-1to1':  
            self.scale = scale
        else:
            raise ValueError(f"Scale: {scale} not supported")
            
        if data_split is None:
            self.data_split = {"train": 0.9, "val": 0.1, "test": 0.0}
        else:
            if not isinstance(data_split, dict):
                raise ValueError("Split must be a dictionary")
            
            if not all(isinstance(i, (int, float)) for i in data_split.values()):
                raise ValueError("All split values must be numbers")
            
            if set(data_split.keys()) != {'train', 'val', 'test'}:
                raise ValueError(f"Split keys must be exactly {{'train', 'val', 'test'}}, got: {set(data_split.keys())}")
               
            if not (0 < data_split["train"] <= 1 and 0 <= data_split['test'] <= 1 and 0 <= data_split['val'] <= 1):
                raise ValueError(f"Split values out of range: train={data_split['train']}, val={data_split['val']}, test={data_split['test']}")
            
            epsilon = 1e-6
            sum_splits = sum(data_split.values())
            
            if not (abs(sum_splits - 1.0) <= epsilon):
                raise ValueError(f"Split values must sum to exactly 1.0 (±{epsilon}). Got sum: {sum_splits}")
            
            self.data_split = data_split.copy()   
            
        self.stratify = stratify
        self.seed = seed
        
        if orientation == 'cols':
            self.orientation = orientation
        else:
            raise ValueError(f"Orientation: {orientation} not supported")
        
        self.center = center
        self.cache_path = cache_path
        self.version = version
        
        if num_features > 0:
            self.num_features = num_features
        else:
            raise ValueError(f"Please check your num_features: {num_features}")
            
        if num_classes > 0:
            self.num_classes = num_classes
        else:
            raise ValueError(f"Please check your num_classes: {num_classes}")
        
        if label_map is None:
            self.label_map = {x: f"{x}" for x in range(num_classes)}
        elif isinstance(label_map, dict):
            if not all(isinstance(key, int) for key in label_map.keys()):
                raise ValueError('All label_map keys must be integers')
            
            if not all(isinstance(val, str) for val in label_map.values()):
                raise ValueError('All label_map values must be strings')
            
            expect_keys = set(range(num_classes))
            actual_keys = set(label_map.keys())
            
            if actual_keys == expect_keys:
                self.label_map = label_map
            else:
                missing = expect_keys - actual_keys
                extra = actual_keys - expect_keys
                error_msg = 'label_map keys mismatch: '
                
                if missing:
                    error_msg += f"missing {missing}"
                
                if extra:
                    error_msg += f" extra {extra}"
                    
                raise ValueError(error_msg)  
            
        else:
            raise ValueError(f"Please check consistency in data on your label_map: {label_map}")
        
        self._pixels_raw = None
        self._labels_raw = None
        self.X = None
        self.y_raw = None
        self.Y = None
        self.idxs = {"train": None, "val": None, "test": None}
        self.summary_ = None
        self._cache_loaded = False
        self.preprocessing_ = {
            "applied": False,
            "scale": self.scale, 
            "center": self.center, 
            "dtype_raw": self.dtype_raw, 
            "orientation": self.orientation, 
            "num_features": self.num_features, 
            "num_classes": self.num_classes,
        }
        
    def load(self):        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"File: {self.dataset_path} doesn't exist. Check your path")
        
        df = pd.read_csv(
                self.dataset_path, 
                header = 0 if self.has_header else None, 
                dtype = self.dtype_raw
            )
        
    
        if not self.expect_cols == df.shape[1]:
            raise ValueError(f"Expected columns: {self.expect_cols}, got: {df.shape[1]}")
        
        if not len(df) > 0:
            raise ValueError(f"Your dataset is empty, rows count: {len(df)}")
        
        self._labels_raw = df.iloc[:, 0].to_numpy(copy=False)
        self._pixels_raw = df.iloc[0:, 1:].to_numpy(copy=False) 
        
        self.N = self._labels_raw.shape[0]
        
        labels = self._labels_raw.shape
        pixels = self._pixels_raw.shape
        
        if not np.isfinite(self._labels_raw).all():
            raise ValueError("Labels contain NaN/Inf")
        
        if not np.isfinite(self._pixels_raw).all():
            raise ValueError("Pixels contain NaN/Inf")
        
        if not np.issubdtype(self._labels_raw.dtype, np.integer):
            raise ValueError(f"Labels must have integer dtype, got: {self._labels_raw.dtype}")
        
        if not ((self._labels_raw >= 0) & (self._labels_raw < self.num_classes)).all():
            raise ValueError(f"All labels must be in range [0, {self.num_classes-1}]")
        
        if not (labels == (len(df), ) and pixels == (len(df), self.num_features)):
            raise ValueError(f"Please check your data shapes: labels shape: {labels} pixels: {pixels}")
    
    def parse_and_shape(self): 
            
        if self._pixels_raw is None or self._labels_raw is None:
            raise RuntimeError("parse_and_shape() requires load() to run first - found _pixels_raw or _labels_raw is None. Call .load() before .parse_and_shape()")
        
        if not isinstance(self._pixels_raw, np.ndarray):
            raise ValueError(f"Expected _pixels_raw to be a NumPy ndarray; got \n{type(self._pixels_raw).__name__}. Verify .load() created NumPy arrays.")
        
        N = self.N
        
        if self._pixels_raw.shape != (N, self.num_features):
            raise ValueError(f"Invalid _pixels_raw shape: expected (N={N}, F={self.num_features}), got {self._pixels_raw.shape}")
        
        if not isinstance(self._labels_raw, np.ndarray):
            raise ValueError(f"Expected _labels_raw to be a NumPy ndarray; got \n{type(self._labels_raw).__name__}. Verify .load() created NumPy arrays.")
        
        if self._labels_raw.shape != (N, ):
            raise ValueError(f"Invalid _labels_raw shape: expected (N={N}, ), got: {self._labels_raw.shape}")
        
        # Remove if you want a generic loader
        if self.num_classes != 10:
            raise ValueError(f"num_classes must be 10 for MNIST (digits 0–9); got {self.num_classes}.")
        
        if self.num_features != 784:
            raise ValueError(f"num_features must be 784 (28×28); got {self.num_features}.")
          
        if self.orientation != 'cols':
            raise ValueError(f"Unsupported orientation '{self.orientation}'. Only 'cols' (samples-as-columns) is supported.") 
        
         # me falta probar este codigo y validarlo si esta bien hecho con chatgpt
        def to_one_hot(y_raw, num_classes):
            if not y_raw.shape == (N, ):
                raise ValueError(f"y_raw must be 1D with length N={N}; got shape {y_raw.shape}.")
                
            if not np.issubdtype(y_raw.dtype, np.integer):
                raise ValueError(f"y_raw must have an integer dtype for indexing; got {y_raw.dtype}.")
                
            if num_classes < 2:
                raise ValueError(f"num_classes must be ≥ 2; got {num_classes}.")
                
            Y = np.zeros((num_classes, N), dtype=np.float32)
            Y[y_raw, np.arange(y_raw.size)] = 1.0
            return Y
            
        X = self._pixels_raw.T
        y_raw = self._labels_raw  
        Y = to_one_hot(y_raw, self.num_classes)
            
        self.X = X
        self.y_raw = y_raw
        self.Y = Y   
        
        # Post conditions para revisar la data generada
        if self.X.shape != (self.num_features, N):
            raise RuntimeError(f"Postcondition failed: X must be shaped (F={self.num_features}, N={N}); got {self.X.shape}.")
    
        if self.y_raw.shape != (N,):
            raise RuntimeError(f"Postcondition failed: y_raw must be shaped (N={N},); got {self.y_raw.shape}.")
    
        if self.Y.shape != (self.num_classes, N):
            raise RuntimeError(f"Postcondition failed: Y must be shaped (C={self.num_classes}, N={N}); got {self.Y.shape}.")
    
        # Check one-hot properties
        if not np.allclose(self.Y.sum(axis=0), 1.0):
            raise RuntimeError("Postcondition failed: each column of Y must sum to 1 (one-hot). Found non-one-hot column(s).")
        
        if not (self.Y.argmax(axis=0) == self.y_raw).all():
            raise RuntimeError("Postcondition failed: Y.argmax(axis=0) must match y_raw exactly.")    
    
    def preprocess(self):
        
        if not isinstance(self.X, np.ndarray):
            raise ValueError(f'preprocess() requires self.X as a NumPy array; got {type(self.X).__name__}.')
        
        N = self.X.shape[1]
        
        if not (self.X.shape == (self.num_features, N) and N > 0):
            raise ValueError(f'X must be shaped (F={self.num_features}, N) with N>0; got {self.X.shape}.')
        
        if not ((self.X >= 0) & (self.X <= 255)).all():
            raise ValueError(f'Raw pixels must be integers in [0,255] before scaling.')
        
        if self.preprocessing_['applied']:
            raise ValueError('Data already appears preprocessed; refusing to run twice.')
            
        if self.scale == "0to1":
            self.X = self.X / 255
        elif self.scale == "-1to1":
            self.X = (self.X - 127.5) / 127.5
        else:
            raise ValueError(f"Unsupported scale '{self.scale}'; expected '0to1' or '-1to1'.")
        
        # I don't know what its supposed to go here
        if self.center:
            pass
        
        self.preprocessing_ = {
            "applied": False,
            "scale": self.scale, 
            "center": self.center, 
            "dtype_raw": self.dtype_raw, 
            "orientation": self.orientation, 
            "num_features": self.num_features, 
            "num_classes": self.num_classes,
            "idempotent_guard": datetime.datetime.now()
        }
       
    def split(self):
        N = self.N
        train = self.data_split['train']
        val = self.data_split['val']
        
        if self.stratify:
            rng = random.Random(self.seed)
            
            # Build class-to-indices mapping and shuffle within each class
            Ic = {c: [idx for idx, label in enumerate(self.y_raw) if label == c] for c in range(self.num_classes)}
            for class_indices in Ic.values():
                rng.shuffle(class_indices)
            
            # Split each class and collect indices
            train_ids, val_ids, test_ids = [], [], []
            
            for c in range(self.num_classes):
                n_c = len(Ic[c])
                t_c = round(train * n_c)
                v_c = round(val * n_c)
                
                train_ids.extend(Ic[c][:t_c])
                val_ids.extend(Ic[c][t_c:t_c + v_c])
                test_ids.extend(Ic[c][t_c + v_c:])
            
            # Shuffle each split to avoid class blocks
            rng.shuffle(train_ids)
            rng.shuffle(val_ids)
            rng.shuffle(test_ids)
            
            self.idxs["train"] = np.array(train_ids, dtype=int)
            self.idxs["val"] = np.array(val_ids, dtype=int) 
            self.idxs["test"] = np.array(test_ids, dtype=int)
            
        else:
            # Simple random split
            rng = random.Random(self.seed)
            indices = list(range(N))
            rng.shuffle(indices)
            
            n_train = round(train * N)
            n_val = round(val * N)
            
            self.idxs["train"] = np.array(indices[:n_train], dtype=int)
            self.idxs["val"] = np.array(indices[n_train:n_train + n_val], dtype=int)
            self.idxs["test"] = np.array(indices[n_train + n_val:], dtype=int)
        
        # Simple post-check
        total = len(self.idxs["train"]) + len(self.idxs["val"]) + len(self.idxs["test"])
        assert total == N, f"Split doesn't sum to N: {total} != {N}"          
            
            
    def get_split(self, name):
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split name '{name}'. Must be one of: 'train', 'val', 'test'")
        
        if self.idxs[name] is None:
            raise RuntimeError(f"Split '{name}' not available. Call .split() first to create data splits.")
        
        if self.X is None or self.Y is None or self.y_raw is None:
            raise RuntimeError("Data not loaded. Call .load() and .parse_and_shape() first.")
        
        indices = self.idxs[name]
        
        X_subset = self.X[:, indices]
        Y_subset = self.Y[:, indices] 
        y_raw_subset = self.y_raw[indices]
        
        return X_subset, Y_subset, y_raw_subset 
    
# DataLoader Pipeline
data = DataLoader('./data/digit-recognizer/train.csv')
data.load()
data.parse_and_shape()
data.preprocess()
data.split()

X_train, Y_train, y_train = data.get_split("train")
X_val, Y_val, y_val = data.get_split("val")
X_test, Y_test, y_test = data.get_split("test")

print(X_train.shape, Y_train.shape, y_train.shape)
print(X_val.shape, Y_val.shape, y_val.shape)
print(X_test.shape, Y_test.shape, y_test.shape)
       
"""
Split policy
------------
`split` must be a dict with exactly the keys {'train','val','test'} and float
fractions in [0,1]. The values must sum to 1.0 within a small tolerance
(±1e-6). `train` must be > 0. This strict policy is intentional to ensure that
*every* sample is assigned to exactly one split (no leftovers, no double-counts),
and to keep runs reproducible.

If `split` is None, the default is {'train': 0.9, 'val': 0.1, 'test': 0.0}.
If `stratify` is True, class proportions are preserved per split. The splitting
(and later batch shuffles) are deterministic given `seed`.

"""      
    

"""
Lo que tengo que hacer para esta clase es:

0. __init__ LISTO

1. load_csv LISTO

2. parse_and_shape LISTO
    2.1 to_one_hot LISTO
    
3. preprocess LISTO

4. split LISTO
5. get_split LISTO

6. batches

7. save_cache
8. load_cache

9. validate
10. summary

11. get_metadata

Basicamente lo que estuve hablando con el chat: https://chatgpt.com/g/g-p-68a624f6af808191b42299d9be37a81f-mnist-nn/c/68a62778-122c-8328-8db0-18f3e49ea414


Love this mindset—if you understand the “why,” the “how” becomes obvious. Here’s what each piece of a solid `DataLoader` is doing, why it matters for your MNIST project, and what “done” looks like.

---

## 1) Load the data

**What:** Read the MNIST CSV into memory. In most CSV versions, each row is one image; the first column is the label (0–9) and the remaining 784 columns are pixel values.

**Why:** Everything else builds on this—no clean load, no clean training.

**Done looks like:** you can access:

* `pixels_raw` with shape `(N, 784)` (or transposed if you choose the “samples as columns” convention).
* `labels_raw` with shape `(N,)` and integer classes `0..9`.

---

## 2) Parse & shape

**What:** Decide and enforce array shapes and orientation throughout the project.

* You already chose **“samples as columns.”** That means use:

  * `X`: `(features, N)` → `(784, N)`
  * `y_raw`: `(N,)`
  * `Y` (one-hot): `(classes, N)` → `(10, N)`

**Why:** Consistency keeps your math and backprop simple. Dense layers expect vectors; MNIST images must be **flattened** (28×28 → 784) for a fully connected net. (Conv nets can keep H×W, but you’re doing FC now.)&#x20;

**Done looks like:** one place (your loader) guarantees these shapes; the rest of the code can assume them.

---

## 3) Preprocess

**What:** Scale pixel values to a small, consistent range (e.g., 0–1 or −1–1); optionally center around 0.

**Why:** Neural nets train better when inputs are in similar, modest ranges. Scaling avoids instabilities and works harmoniously with common activations (ReLU, sigmoid, softmax). For images, dividing by 255 (0–1) or shifting to −1–1 are both common.  &#x20;

**Important rule:** Fit your scaling **using training data** and apply the *same* scaler to validation/test and to future inference inputs. Don’t let test data influence preprocessing choices (no leakage). &#x20;

**Done looks like:** you store the scaling recipe (e.g., “divide by 255” or min/max/mean), and `X` is scaled accordingly.

---

## 4) Splitting & reproducibility

**What:** Split into train/validation(/test) and make the split repeatable (set a random seed).

**Why:** You need **out-of-sample** data to detect overfitting and measure generalization. A repeatable split helps debugging and comparisons.&#x20;

**Done looks like:** deterministic masks/indices for `train`, `val`, (`test`), with a saved `random_state` so you can recreate the split later.

---

## 5) Batching

**What:** Serve data in mini-batches (e.g., 64 or 128 samples at a time) and shuffle the order each **epoch**.

**Why:** Batches make training efficient and stable; shuffling prevents the model from seeing long runs of a single class and becoming biased. When shuffling, keep samples and labels aligned. &#x20;

**Epochs?** One **epoch** = the model has seen every training sample once (usually via many batches). You’ll see loss/accuracy reported per step and per epoch.&#x20;

**Done looks like:** an iterator that, per epoch, shuffles consistently, yields `(X_batch, Y_batch)` (and `y_raw_batch` if you want), and handles the final “short” batch gracefully.

---

## 6) Caching

**What:** Save expensive-to-recompute artifacts (e.g., scaled `X`, one-hot `Y`, split indices) to disk, typically in a fast NumPy format.

**Why:** Faster dev loop. You avoid re-parsing CSV and re-doing preprocessing every run. It also reduces the chance of accidental drift in preprocessing across runs.

**Done looks like:** a small `.npz` that stores `{X, y_raw, Y, split_indices, scaler_metadata, version/tag}` and a simple “is cache valid?” check (same CSV file, same preprocessing settings).

---

## 7) Sanity checks & diagnostics

**What:** Quick validations that catch subtle data bugs *before* training:

* **Shapes/orientation:** `X.shape == (784, N)`, `Y.shape == (10, N)`, `y_raw.shape == (N,)`.
* **Ranges:** min/max of `X` match your scaling (e.g., 0–1 or −1–1).&#x20;
* **Label alignment after shuffle:** randomly inspect a few images and confirm the label matches. (Shuffling samples and labels independently is a classic pitfall.) &#x20;
* **Class balance:** histogram of labels—MNIST is roughly balanced; if yours isn’t, you’ll want to know.
* **No NaNs/Infs:** check for invalid numbers.
* **Tiny overfit test:** train on a very small subset and confirm the model can drive loss near zero—if it can’t, your pipeline might be broken.

**Why:** These catch the mistakes that silently ruin training.

**Done looks like:** a `DataLoader.validate()` step that prints a short report (shapes, ranges, class counts) and maybe shows a few sample images/labels.

---

## 8) Metadata for inference

**What:** Save everything you’ll need to reproduce preprocessing and interpret predictions later:

* Input spec: “expects `(784,)` vector scaled by **\\[your rule]**; orientation = samples-as-columns.”
* Label mapping `0..9` → class names (if you’re using names).
* One-hot vs sparse choice (your training used one-hot targets—store that).
* The scaler parameters or the exact rule (e.g., “divide by 255” or “subtract 127.5 then divide 127.5”). You *must* apply the same to user-drawn digits in your React app.&#x20;

**Why:** The model is only as good as the data you feed it—at inference time you must **mirror** training-time preprocessing or predictions will degrade.&#x20;

**Done looks like:** alongside saved weights, you persist a small JSON (or similar) with the above fields and the random seed used for splits/shuffles.

---

### TL;DR — what your `DataLoader` should *return/expose*

* **Tensors/arrays:** `X_train, Y_train, y_train_raw`, `X_val, Y_val, y_val_raw`, (`X_test, ...`).
* **Iterators:** a batched, shuffling iterator over the train split (and non-shuffling for val/test).
* **Metadata:** scaler rule/params, label mapping, input shape/orientation, random seed, and any cache/version info.
* **Diagnostics:** quick stats (shapes, ranges, class counts) and a simple `validate()` method.

If you’d like, we can turn this into a short acceptance checklist you can keep next to `data_loader.py` so you can tick each item off as you implement it.



"""    
   
# DESCRIPCIONES DE FUNCIONES     
        
"""
Awesome—let’s turn your list into a **crisp, code-ready API blueprint**. No code here, just exact function contracts so you can implement confidently.

---

# 1) Load the data

### `load_csv(path, *, expect_cols=785, has_header=True, dtype="uint8") -> None`

* **Purpose:** Read raw CSV bytes into memory safely.
* **Needs to work:** a file path; optional header flag; expected column count (1 label + 784 pixels).
* **Params:**

  * `path`: string to `mnist.csv`
  * `expect_cols`: sanity guard (should be 785)
  * `has_header`: skip header row if present
  * `dtype`: read pixels as `uint8` initially (0–255)
* **Side effects / internal process:**

  * Open file, parse into a NumPy array.
  * Split first column → internal `self._labels_raw (N,)`, remaining → `self._pixels_raw (N, 784)`.
  * Assert label range in `{0..9}`; assert pixel range in `[0,255]`.
* **Returns:** nothing (stores raw arrays internally).

> Later you can add `load_idx(images_path, labels_path)` that populates the same internals. Dense nets need flattened vectors, so CSV with 784 columns is already “flat.” If you ever start from 28×28, you’ll flatten here (28×28 → 784) for FC nets.&#x20;

---

# 2) Parse & shape

### `parse_and_shape(orientation="cols") -> None`

* **Purpose:** Commit to the project-wide shapes/orientation.
* **Needs to work:** `_pixels_raw (N, 784)`, `_labels_raw (N,)`.
* **Params:**

  * `orientation`: `"cols"` means **samples-as-columns**
* **Internal process:**

  * Transpose pixels to `X: (784, N)` if `orientation=="cols"`.
  * Store `y_raw: (N,)`.
  * Create `Y: (10, N)` via one-hot (see helper below).
* **Returns:** nothing (sets `self.X`, `self.y_raw`, `self.Y`).

### Helper: `to_one_hot(y_raw, num_classes=10, orientation="cols") -> Y`

* **Purpose:** One-hot encode once, consistently.
* **Returns:** `Y (10, N)` for `orientation=="cols"`.

---

# 3) Preprocess

### `preprocess(scale="0to1", *, center=False) -> None`

* **Purpose:** Put pixels in a friendly numeric range for training.
* **Needs to work:** `self.X` present.
* **Params:**

  * `scale`: `"0to1"` (divide by 255) or `"-1to1"` (subtract 127.5 then divide 127.5)
  * `center`: optional mean-centering (off by default)
* **Internal process:**

  * Cast to `float32`.
  * Apply chosen scaling; optionally subtract mean (if you enable center).
  * Record the **exact recipe** in `self.preprocessing_` (for reproducibility/inference).
* **Returns:** nothing.
* **Why:** NNs train more stably with small, consistent ranges; many activations behave best in ≈\\[0,1] or \\[−1,1].  &#x20;

---

# 4) Splitting & reproducibility

### `split(train=0.9, val=0.1, test=0.0, *, stratify=True, seed=42) -> None`

* **Purpose:** Create deterministic indices for each split.
* **Needs to work:** `y_raw (N,)`.
* **Params:**

  * `train`, `val`, `test`: fractions that sum to ≤ 1.0
  * `stratify`: keep class proportions similar across splits
  * `seed`: reproducible permutation
* **Internal process:**

  * Build a permutation with `seed`.
  * If `stratify`, permute within each class then interleave.
  * Slice indices into `self.idxs = {"train":…, "val":…, "test":…}`.
  * Persist `seed` and the final index arrays.
* **Returns:** nothing.
* **Why:** You need held-out data to check generalization (avoid overfitting), and the ability to reproduce a split for debugging.&#x20;

### `get_split(name: Literal["train","val","test"]) -> (X, Y, y_raw)`

* **Purpose:** Provide ready-to-train arrays.
* **Returns:** views (not copies if possible) with shapes `(784, n)`, `(10, n)`, `(n,)`.

---

# 5) Batching

### `batches(split="train", batch_size=128, *, shuffle=True, drop_last=False) -> Iterator[(Xb, Yb, yb)]`

* **Purpose:** Feed the net mini-batches with consistent shapes.
* **Needs to work:** `self.idxs[split]` and `self.X/Y/y_raw`.
* **Params:**

  * `shuffle`: `True` for train (reshuffle **each epoch**), `False` for val/test
  * `drop_last`: if the last incomplete batch should be dropped
* **Internal process:**

  * For train: create a new permuted view (deterministic if you pass a step/epoch seed).
  * Slice contiguous blocks of indices; yield `(784,B)`, `(10,B)`, `(B,)`.
* **Returns:** generator/iterator.

*(One epoch = you iterate all batches that cover the whole split once.)*

---

# 6) Caching

### `save_cache(path="data/processed/mnist.npz", *, version="v1") -> None`

* **Purpose:** Avoid re-parsing/scaling every run.
* **Needs to work:** arrays + metadata ready.
* **Internal process:**

  * Save: `X`, `Y`, `y_raw`, `idxs`, `preprocessing_`, `orientation`, `label_map`, `seed`, `version`.
  * Optionally store a checksum of the raw CSV and the settings used.

### `load_cache(path="data/processed/mnist.npz", *, require_version=None) -> bool`

* **Purpose:** Load preprocessed arrays if compatible.
* **Params:**

  * `require_version`: invalidate cache if version mismatches
* **Returns:** `True` if successfully loaded and validated, else `False`.
* **Why:** Faster dev loop; fewer accidental diffs in preprocessing between runs.

---

# 7) Sanity checks & diagnostics

### `validate() -> dict`

* **Purpose:** Catch silent data bugs *before* training.
* **Internal process / checks:**

  * Shapes/orientation: `X (784,N)`, `Y (10,N)`, `y_raw (N,)`.
  * Ranges after preprocessing: min/max ≈ expected (e.g., 0..1 or −1..1).&#x20;
  * Class histogram (roughly balanced for MNIST).
  * No NaNs/Infs.
  * Optional: spot-check a few `(image,label)` pairs (label alignment after shuffles).
* **Returns:** summary dict (counts, ranges, dtypes), so `train.py` can print/log it.

### `summary() -> dict`

* **Purpose:** Lightweight snapshot for logs.
* **Returns:** `{n_total, splits, class_counts, X_dtype, y_dtype, X_range, X_shape, scale_mode, seed, cache_path}`.

---

# 8) Metadata for inference

### `get_metadata() -> dict`

* **Purpose:** Give the FastAPI endpoint everything it needs to preprocess user input identically.
* **Returns:** e.g.

  ```text
  {
    "input_shape": [784],               # expected vector length
    "orientation": "cols",              # samples-as-columns
    "num_classes": 10,
    "label_map": {0:"0",...,9:"9"},     # or names if you had them
    "preprocessing": {
       "scale": "0to1",                 # or "-1to1"
       "center": false
    },
    "seed": 42,
    "version": "v1"
  }
  ```
* **Why:** Inference must mirror training preprocessing exactly, or accuracy will drop.&#x20;

---

## Handy private helpers (optional but useful)

* `_assert_label_range(min_=0, max_=9) -> None`
* `_compute_class_indices(y_raw) -> dict[int, np.ndarray]` (for stratified split)
* `_shuffle_in_unison(index_array, rng) -> None`
* `_check_cache_compat(config) -> bool`
* `_preview_sample(idx) -> (img_28x28, label)` (used by your visualize script)

---

### Acceptance checklist (what “done” means)

* Loader can **ingest CSV**, produce `X/Y/y_raw` with **correct shapes**.
* Preprocessing is **documented + reproducible** (metadata stored).
* Split is **deterministic** with a seed; **stratified** by class.
* Batching yields **(784,B)** and **(10,B)** consistently.
* Cache round-trips work and are **validated**.
* `validate()` returns clean diagnostics; no NaNs, ranges as expected.
* `get_metadata()` is enough for your FastAPI endpoint to preprocess a canvas image identically to training.

If you want, I can help you convert this into docstrings you can paste on each method while you implement.


"""