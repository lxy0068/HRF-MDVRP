# HRF-MDVRP
This is our implementation for the paper: Hyper-Relation Fusion for Solving Multi-depot  Vehicle Routing Problems
![frame (2)](https://github.com/user-attachments/assets/f3c38644-9820-424c-be40-360ce355f0a1)
# Abstract
Multi-Depot Vehicle Routing Problem (MDVRP) requires constructing routes from multiple depots to geographically dispersed customers under capacity constraints. 
Unlike single-depot routing problems, MDVRP requires determining not only the routing relationship between customers but also the assignment relationship of customers to depots.
In this paper, we propose a Hyper-Relation Fusion (HRF) neural combinatorial optimization algorithm to solve MDVRP, considering both heterogeneous relationships and homogeneous relationships between depots and customers. The heterogeneous relationships of depot-customer and customer-customer are captured through graph attention to distinguish different types of connectivity. The homogeneous relationships are learned by aggregating the features of all nodes via a graph convolutional network. Finally, HRF fuses the original node features, heterogeneous features, and homogeneous features, which are further processed through an encoder-decoder architecture to generate the solution. Comprehensive experiments on synthetic and benchmark datasets demonstrate that HRF surpasses the state-of-the-art metaheuristics and learning-based methods in solution quality.
# Setup
The released code consists of the following files.

# Dependencies

# Usage
