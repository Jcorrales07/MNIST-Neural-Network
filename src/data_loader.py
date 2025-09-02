import pandas as pd
import numpy as np
import os
import random

from neural_network.layers import DenseLayer

class DataLoader:
    """
    Simple MNIST DataLoader with column-major orientation (features, samples).
    
    Features:
    - Loads MNIST CSV data
    - Scales pixels to [0,1] or [-1,1]
    - Stratified train/val/test splits
    - Batch generation
    - One-hot encoding
    """
    
    def __init__(self, dataset_path, scale="0to1", data_split=None, seed=42):
        self.dataset_path = dataset_path
        self.scale = scale  # "0to1" or "-1to1"
        self.seed = seed
        
        # Default split: 80% train, 10% val, 10% test
        if data_split is None:
            self.data_split = {"train": 0.8, "val": 0.1, "test": 0.1}
        else:
            if abs(sum(data_split.values()) - 1.0) > 1e-6:
                raise ValueError(f"Split values must sum to 1.0, got {sum(data_split.values())}")
            self.data_split = data_split
        
        # Data storage
        self.X = None           # (784, N) - pixels as columns
        self.Y = None           # (10, N) - one-hot labels as columns  
        self.y_raw = None       # (N,) - raw integer labels
        self.N = None           # number of samples
        self.splits = {"train": None, "val": None, "test": None}
        
    def load_and_process(self):
        """Load CSV, preprocess, split - all in one step."""
        
        # 1. Load CSV
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        df = pd.read_csv(self.dataset_path)
        
        if df.shape[1] != 785:  # 1 label + 784 pixels
            raise ValueError(f"Expected 785 columns (1 label + 784 pixels), got {df.shape[1]}")
        
        self.N = len(df)
        print(f"Loaded {self.N} samples")
        
        # 2. Extract labels and pixels
        labels = df.iloc[:, 0].values  # First column = labels
        pixels = df.iloc[:, 1:].values # Rest = pixel values
        
        # Basic validation
        if not ((labels >= 0) & (labels <= 9)).all():
            raise ValueError("Labels must be digits 0-9")
        if not ((pixels >= 0) & (pixels <= 255)).all():
            raise ValueError("Pixels must be in range 0-255")
        
        # 3. Scale pixels
        if self.scale == "0to1":
            pixels = pixels / 255.0
        elif self.scale == "-1to1":
            pixels = (pixels - 127.5) / 127.5
        else:
            raise ValueError(f"Scale must be '0to1' or '-1to1', got '{self.scale}'")
        
        # 4. Convert to column-major format
        self.X = pixels.T  # Shape: (784, N)
        self.y_raw = labels
        
        # 5. Create one-hot encoding
        self.Y = np.zeros((10, self.N))
        self.Y[labels, np.arange(self.N)] = 1.0  # Shape: (10, N)
        
        # 6. Create stratified splits
        self._create_splits()
        
        print(f"Data processed: X={self.X.shape}, Y={self.Y.shape}")
        print(f"Splits - train: {len(self.splits['train'])}, val: {len(self.splits['val'])}, test: {len(self.splits['test'])}")
        
    def _create_splits(self):
        """Create stratified train/val/test splits."""
        rng = random.Random(self.seed)
        
        # Group indices by class
        class_indices = {c: [] for c in range(10)}
        for idx, label in enumerate(self.y_raw):
            class_indices[label].append(idx)
        
        # Shuffle within each class
        for indices in class_indices.values():
            rng.shuffle(indices)
        
        # Split each class proportionally
        train_ids, val_ids, test_ids = [], [], []
        
        for class_idx, indices in class_indices.items():
            n = len(indices)
            n_train = round(self.data_split["train"] * n)
            n_val = round(self.data_split["val"] * n)
            
            train_ids.extend(indices[:n_train])
            val_ids.extend(indices[n_train:n_train + n_val])
            test_ids.extend(indices[n_train + n_val:])
        
        # Final shuffle to mix classes
        rng.shuffle(train_ids)
        rng.shuffle(val_ids) 
        rng.shuffle(test_ids)
        
        self.splits = {
            "train": np.array(train_ids),
            "val": np.array(val_ids),
            "test": np.array(test_ids)
        }
    
    def get_split(self, name):
        """Get data for a specific split."""
        if name not in self.splits:
            raise ValueError(f"Invalid split '{name}'. Use: train, val, test")
        
        if self.X is None:
            raise RuntimeError("Call load_and_process() first")
        
        indices = self.splits[name]
        return (
            self.X[:, indices],      # (784, n_split)
            self.Y[:, indices],      # (10, n_split) 
            self.y_raw[indices]      # (n_split,)
        )
    
    def batches(self, split="train", batch_size=128, shuffle=True):
        """Generate batches for training/evaluation."""
        if self.X is None:
            raise RuntimeError("Call load_and_process() first")
        
        indices = self.splits[split].copy()
        
        if shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        # Generate batches
        n_samples = len(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            yield (
                self.X[:, batch_indices],      # (784, batch_size)
                self.Y[:, batch_indices],      # (10, batch_size)
                self.y_raw[batch_indices]      # (batch_size,)
            )
    
    def summary(self):
        """Print data summary."""
        if self.X is None:
            print("No data loaded")
            return
        
        print(f"\n=== MNIST Data Summary ===")
        print(f"Total samples: {self.N}")
        print(f"Data shape: X={self.X.shape}, Y={self.Y.shape}")
        print(f"Pixel range: [{self.X.min():.3f}, {self.X.max():.3f}]")
        print(f"Scale mode: {self.scale}")
        print(f"Random seed: {self.seed}")
        
        print(f"\nSplit sizes:")
        for name, indices in self.splits.items():
            print(f"  {name}: {len(indices)} samples ({len(indices)/self.N*100:.1f}%)")
        
        print(f"\nClass distribution:")
        unique, counts = np.unique(self.y_raw, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count/self.N*100:.1f}%)")


# Example usage:
if __name__ == "__main__":
    # Initialize loader
    loader = DataLoader(
        dataset_path="./data/digit-recognizer/train.csv",
        scale="0to1",
        data_split={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42
    )
    
    # Load and process everything
    loader.load_and_process()
    
    # Show summary
    loader.summary()
    
    # Get training data
    X_train, Y_train, y_train = loader.get_split("train")
    print(f"\nTraining data: X={X_train.shape}, Y={Y_train.shape}")
    
    capa = DenseLayer(784, 64)
    
    # Generate batches
    print(f"\nFirst 3 training batches:")
    for i, (X_batch, Y_batch, y_batch) in enumerate(loader.batches("train", batch_size=32)):
        print(f"  Batch {i+1}: X={X_batch.shape}, Y={Y_batch.shape}, y={y_batch.shape}")
        
        # Implementacion de la capa, y el uso de forward
        capa.forward(X_batch)
        print(capa.output)
        
        if i >= 2:  # Just show first 3
            break