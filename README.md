# HRF-MDVRP
This is our implementation for the paper: **Hyper-Relation Fusion for Solving Multi-depot  Vehicle Routing Problems**
![frame (2)](https://github.com/user-attachments/assets/f3c38644-9820-424c-be40-360ce355f0a1)
# Abstract
Multi-Depot Vehicle Routing Problem (MDVRP) requires constructing routes from multiple depots to geographically dispersed customers under capacity constraints. 
Unlike single-depot routing problems, MDVRP requires determining not only the routing relationship between customers but also the assignment relationship of customers to depots.
In this paper, we propose a Hyper-Relation Fusion (HRF) neural combinatorial optimization algorithm to solve MDVRP, considering both heterogeneous relationships and homogeneous relationships between depots and customers. The heterogeneous relationships of depot-customer and customer-customer are captured through graph attention to distinguish different types of connectivity. The homogeneous relationships are learned by aggregating the features of all nodes via a graph convolutional network. Finally, HRF fuses the original node features, heterogeneous features, and homogeneous features, which are further processed through an encoder-decoder architecture to generate the solution. Comprehensive experiments on synthetic and benchmark datasets demonstrate that HRF surpasses the state-of-the-art metaheuristics and learning-based methods in solution quality.
# Setup
The released code consists of the following files.

# Dependencies
**Python 3.8+ required**  
Install PyTorch and PyTorch Geometric with CUDA 11.3 support:  
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```
# Usage
### Training the Model

Train a new model for MDVRP or MDOVRP:

```bash
# Train MDVRP model
python MDVRP_Train.py

# Train MDOVRP model 
python MDOVRP_Train.py
```

### Evaluating the Model

Evaluate a trained model on test instances:

```bash
# Evaluate MDVRP model
python MDVRP_Eval.py

# Evaluate MDOVRP model
python MDOVRP_Eval.py
```

## Configuration

All parameters are configurable directly in the respective `.py` files. Key modifiable parameters include:

• **Training Parameters**: Batch size, learning rate, training epochs

• **Model Architecture**: Hidden dimensions, attention heads

• **Problem Settings**: Number of depots, customers, and vehicles

Example configuration snippet from `MDVRP_Train.py`:

```python
# Hyperparameters
self.batch_size = 512
self.lr = 1e-4
self.n_epochs = 100
self.hidden_dim = 128
```

## Repository Structure

```
├── MDVRP_Train.py       # Training script for MDVRP
├── MDVRP_Eval.py        # Evaluation script for MDVRP
├── MDOVRP_Train.py      # Training script for MDOVRP
├── MDOVRP_Eval.py       # Evaluation script for MDOVRP
└── README.md
```

## License

This project is released for academic/research use. For commercial use, please contact the authors.
