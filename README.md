# FAB-FL: Client Selection Optimization for Federated Learning

## Overview
FAB-FL (Federated Aggregation and Balancing for Federated Learning) is a research-focused simulation framework designed to optimize client selection in distributed machine learning environments. Standard Federated Learning frameworks (like `FedAvg`) often suffer from performance bottlenecks when deployed across heterogeneous clients with varying data distributions (Non-IID) and computational capabilities. 

This repository provides an algorithmic pipeline that evaluates available clients dynamically using a multi-criteria reward function, selecting the optimal client cohort in each communication round to accelerate model convergence and mitigate straggler effects.

---

## Key Features
* **Dynamic Client Selection:** Evaluates clients based on localized performance metrics and computational constraints before aggregation.
* **Balanced Reward Function:** Implements a mathematically rigorous reward mechanism balancing local model quality against computational latency.
* **Non-IID Data Simulation:** Features built-in Dirichlet distribution partitioning tools to simulate real-world data heterogeneity using the CIFAR-10 dataset.
* **Distributed CPU Architecture:** Scalable processing utilizing the Ray distributed framework, optimized for local multi-core simulation and seamless migration to CPU clusters.

---

## Mathematical Foundation
The selection framework ranks participating clients by computing a composite reward score ($R_i$) for each client $i$:

$$R_i = \alpha(\text{Accuracy}_i) + \beta(\text{Latency}_i)^{-1}$$

Where:
* $\text{Accuracy}_i$ represents the validation accuracy achieved during local training.
* $\text{Latency}_i$ is the simulated computational and networking overhead associated with the node.
* $\alpha$ and $\beta$ are configurable hyperparameters controlling the selection priority.

---

## Project Structure
```text
├── data_utils.py  # CIFAR-10 downloading and Non-IID Dirichlet distribution splitting
├── client.py      # Local training loop routine and deterministic latency simulation
├── selector.py    # FAB-FL selection logic, reward evaluation, and client ranking
├── server.py      # Global aggregation loop, server orchestration, and Ray cluster initialization
└── requirements.txt
