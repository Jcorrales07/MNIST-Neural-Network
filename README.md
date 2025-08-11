
## Directory Layout

```
neural_network_project/
├── data/
│   ├── mnist/                    # MNIST dataset files
│   └── models/                   # Saved model weights
├── src/
│   ├── neural_network/
│   │   ├── __init__.py
│   │   ├── layers.py            # Dense layer implementation
│   │   ├── activations.py       # Sigmoid, ReLU, Tanh, Softmax
│   │   ├── losses.py            # Cross-entropy loss
│   │   ├── optimizers.py        # SGD, Adam (later)
│   │   ├── network.py           # Main neural network class
│   │   └── utils.py             # Helper functions
│   ├── data_loader.py           # MNIST loading and preprocessing
│   ├── train.py                 # Training script
│   ├── test.py                  # Testing/evaluation script
│   └── visualize.py             # Plotting loss, predictions
├── backend/                     # FastAPI (Phase 2)
├── frontend/                    # React app (Phase 3)
├── notebooks/                   # Jupyter notebooks for experimentation
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```