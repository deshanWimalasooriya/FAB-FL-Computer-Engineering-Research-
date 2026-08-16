# FAB-FL: Frequency-Aware Bandwidth Federated Learning

<div align="center">
  <h3>A Joint Optimization Framework for Non-IID Data and Resource-Constrained Edge Devices</h3>
</div>

---

## Abstract

Federated Learning (FL) enables distributed model training across edge devices without centralizing private data. However, the presence of **Non-Independent and Identically Distributed (Non-IID)** data and **heterogeneous system resources** (e.g., varying bandwidth and compute speeds) significantly degrades model convergence and introduces severe straggler bottlenecks. 

**FAB-FL** (Frequency-Aware Bandwidth Federated Learning) introduces a novel **Upper Confidence Bound (UCB)** based client selection mechanism. By jointly optimizing for data distribution diversity (**FREQSEL**) and device hardware capabilities (**BSFL**), the framework accelerates convergence, mitigates model bias, and gracefully handles resource constraints.

---

## 1. System Architecture

The FAB-FL framework is built on a modern, decoupled client-server architecture utilizing **FastAPI** for high-throughput orchestrations, **PyTorch** for CUDA-accelerated local training, and **CustomTkinter** for an interactive, glassmorphism-styled visual monitoring interface.

### 1.1 The Global Server Node
- **Orchestration:** Manages global state, registration handshakes, and round synchronization via asynchronous REST endpoints.
- **Client Selection:** Computes joint UCB scores dynamically per round to select a subset of $m$ optimal clients from $N$ registered devices.
- **Aggregation:** Executes the `FedAvg` algorithm to aggregate the locally trained PyTorch `.pt` model weights into a singular global model.

### 1.2 The Local Edge Clients
- **Virtual Simulation:** A single physical client node simulates multiple virtual edge devices, each maintaining a persistent memory state.
- **Accelerated Inference & Training:** Fully leverages local **NVIDIA CUDA** environments with pinned memory (`pin_memory=True`) to rapidly perform local stochastic gradient descent.
- **Data Skew Emulation:** Implements strict Dirichlet distribution partitioning to forcefully simulate real-world Non-IID label skew across virtual nodes.

---

## 2. Mathematical Formulation & Workflow

The core methodology of FAB-FL relies on mathematically quantifying the utility of selecting a client based on its data richness and its hardware speed.

### 2.1 Non-IID Dirichlet Partitioning
To simulate Non-IID data environments realistically on the CIFAR-10 dataset, we sample label distributions from a Dirichlet distribution. For each class $c$, the allocation of samples across $N$ clients is determined by a probability vector $\vec{p}_c$:

$$ \vec{p}_c \sim Dir(\alpha \cdot \vec{1}) $$

Where $\alpha$ acts as the concentration parameter. A smaller $\alpha$ (e.g., $0.1$) leads to a highly heterogeneous, strictly partitioned Non-IID space, while a larger $\alpha$ trends toward uniform IID distributions.

### 2.2 FREQSEL: Frequency-Aware Data Utility
To mitigate bias, the server calculates the pairwise distance of class distributions. Let $C_i$ represent the class frequency vector of client $i$. The data distance $D_i$ measures how divergent client $i$'s local data is from the optimal uniform distribution. We convert this to a normalized utility metric $U_i$, where lower distance implies higher data utility:

$$ U_i = \text{MinMaxNorm}(1.0 - D_i) $$

### 2.3 BSFL: Bandwidth and Speed Normalization
Stragglers disrupt the synchronized aggregation in Federated Learning. The server tracks the computational latency/bandwidth speed $S_i$ (in milliseconds) for each client. The system favors faster clients by computing the inverse speed and normalizing it:

$$ V_i = \text{MinMaxNorm}\left(\frac{1}{S_i + \epsilon}\right) $$

### 2.4 Joint UCB Selection Policy
To avoid greedy algorithms permanently starving clients with unique but slow data, FAB-FL employs the **Upper Confidence Bound (UCB)** algorithm. This perfectly balances **exploitation** (selecting fast clients with good data) and **exploration** (selecting rarely-used clients to ensure fairness).

For round $t$, the UCB score $Q_i(t)$ for client $i$ is calculated as:

$$ \text{Exploitation}_i = \lambda_1 U_i + \lambda_2 V_i $$

$$ \text{Exploration}_i = c \sqrt{\frac{\ln(t)}{N_k(i) + \epsilon}} $$

$$ Q_i(t) = \text{Exploitation}_i + \text{Exploration}_i $$

Where:
- $\lambda_1, \lambda_2$ are weighting hyperparameters for data utility and hardware speed.
- $N_k(i)$ tracks how many times client $i$ has been selected in past rounds.
- $c$ controls the degree of exploration.

In each communication round, the server sorts $Q_i(t)$ and dynamically selects the top $m$ clients to participate.

---

## 3. Workflow Execution

The practical execution of the methodology follows a strict 5-step loop during operation:

1. **Initialization:** The Edge Node generates $N$ virtual clients, downloads CIFAR-10, partitions the data via the $Dir(\alpha)$ distribution, and registers them via HTTP handshakes with the Central Server.
2. **Scoring:** The Central Server calculates $Q_i(t)$ for all registered clients and broadcasts the top $m$ IDs.
3. **Distribution:** The selected local clients download the latest global PyTorch model (`local_global_model.pt`) from the server.
4. **Local Epochs:** Each selected client trains the model on its private subset using Cross-Entropy Loss and SGD, fully offloading matrix multiplications to the CUDA hardware.
5. **FedAvg:** After all $m$ models are uploaded back to the server, they are averaged to generate the new global state, and the cycle repeats.

---

> **Note:** This repository features a fully interactive visual dashboard for both Server and Client operations, featuring real-time graphical progress monitoring and UCB score diagnostics.
