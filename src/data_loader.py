

import pandas as pd
import numpy as np
import random
import os

class DataLoader:
    def __init__(self, dataset_path, has_header: bool = True, expect_cols: int = 785, dtype_raw: str = "uint8", scale: str = "0to1", center: bool = False, split: dict = None, stratify: bool = True, seed: int = 42, orientation: str = "cols", cache_path: str | None = "data/processed/mnist.npz", version: str = "v1", num_features: int = 784, num_classes: int = 10, label_map: dict | None = None):
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
            
        if split is None:
            self.split = {"train": 0.9, "val": 0.1, "test": 0.0}
        else:
            if not isinstance(split, dict):
                raise ValueError("Split must be a dictionary")
            
            if not all(isinstance(i, (int, float)) for i in split.values()):
                raise ValueError("All split values must be numbers")
            
            if set(split.keys()) != {'train', 'val', 'test'}:
                raise ValueError(f"Split keys must be exactly {{'train', 'val', 'test'}}, got: {set(split.keys())}")
               
            if not (0 < split["train"] <= 1 and 0 <= split['test'] <= 1 and 0 <= split['val'] <= 1):
                raise ValueError(f"Split values out of range: train={split['train']}, val={split['val']}, test={split['test']}")
            
            epsilon = 1e-6
            sum_splits = sum(split.values())
            
            if not (abs(sum_splits - 1.0) <= epsilon):
                raise ValueError(f"Split values must sum to exactly 1.0 (±{epsilon}). Got sum: {sum_splits}")
            
            self.split = split.copy()   
            
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
            "scale": self.scale, 
            "center": self.center, 
            "dtype_raw": self.dtype_raw, 
            "orientation": self.orientation, 
            "num_features": self.num_features, 
            "num_classes": self.num_classes,
        }
        
    def load(self):
        df = pd.read_csv(self.dataset_path) 
        self.dataset = np.array(df)
        np.random.shuffle(self.dataset)
        
        self.rows = self.dataset.shape[0]
        self.columns = self.dataset.shape[1]  
        
    def get_split(self, type):
        if type == 'dev':
            data_dev = self.dataset[0:1000].T
            
            X_dev = data_dev[0]
            Y_dev = data_dev[1:self.columns]
            
            X_dev = X_dev / 255
            
            return X_dev, Y_dev
        elif type == "train":
            data_train = self.dataset[1000:self.rows].T
            
            X_train = data_train[0]
            Y_train = data_train[1:self.columns]
            
            X_train = X_train / 255
            
            return X_train, Y_train
            
        
       
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
        
                   
                    
        
        
data = DataLoader('./data/digit-recognizer/train.csv', split={"train": 1.1, "val": -0.1, "test": 0.0})
data.load()
X_train, Y_train = data.get_split('train')

print(data.split)
print(X_train.shape, Y_train.shape)

# Tengo que conocer como se mira el X/Y_train y X/Y_dev 
# Cuales son las funciones minimas para el data loader?
    # Maneja el load dataset
    # Maneja el split de la data en X/Y_train y X/Y_dev 
    # y manejalos batches
    # Eso seria lo minimo no?
    
    

"""
Lo que tengo que hacer para esta clase es:

0. __init__ LISTO

1. load_csv

2. parse_and_shape
    2.1 to_one_hot
    
3. preprocess

4. split
5. get_split

6. batches

7. save_cache
8. load_cache

9. validate
10. summary

11. get_metadata

Basicamente lo que estuve hablando con el chat: https://chatgpt.com/g/g-p-68a624f6af808191b42299d9be37a81f-mnist-nn/c/68a62778-122c-8328-8db0-18f3e49ea414

"""    
        