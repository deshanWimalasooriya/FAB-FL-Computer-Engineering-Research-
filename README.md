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

The core methodology of FAB-FL relies on mathematically quantifying the utility of selecting a client based on its data richness and its hardware speed, ensuring robust convergence even under severe data heterogeneity.

### 2.1 Data Partitioning (Non-IID Generation)
To simulate a real-world, highly skewed non-IID environment on the CIFAR-10 dataset, we partition the data across clients using a Dirichlet distribution. Let $C = 10$ be the total number of classes, and $N$ be the total number of clients. For each class $c \in \{1, \dots, C\}$, we sample a probability vector $p_c$ that dictates how the samples of class $c$ are distributed across the $N$ clients:

$$ p_c = (p_{c,1}, p_{c,2}, \dots, p_{c,N}) \sim \text{Dir}(\alpha \mathbf{1}_N) $$

Given the hyperparameter $\alpha = 0.5$, the probability density function for the Dirichlet distribution of class $c$ across $N$ clients is:

$$ f(p_c; \alpha) = \frac{\Gamma(N\alpha)}{\Gamma(\alpha)^N} \prod_{i=1}^N p_{c,i}^{\alpha-1} $$

- $p_{c,i}$ represents the exact proportion of available samples of class $c$ that are allocated to client $i$.
- $\alpha = 0.5$ creates a heavily skewed distribution, simulating severe data heterogeneity.
- The constraint $\sum_{i=1}^N p_{c,i} = 1$ ensures all data for class $c$ is assigned.

### 2.2 FREQSEL Candidate Filtering (With Penalty)
The goal of FREQSEL is to filter the total pool of $N$ clients down to an initial candidate pool $K$ ($|K| < N$) by selecting clients whose local data distributions are closest to the global distribution $Q$. We measure the distance using the Kullback-Leibler (KL) Divergence:

$$ D_{KL}(P_i \parallel Q) = \sum_{c=1}^C P_i(c) \log\left(\frac{P_i(c)}{Q(c)}\right) $$

To prevent "Class Starvation" mathematically, we minimize this divergence across the selected clients while adding a heavy penalty. The initial candidate pool $K$ is the subset $S \subset \{1, \dots, N\}$ that minimizes:

$$ K = \arg\min_{\substack{S \subset \{1..N\} \\ |S|=k}} \left( \sum_{i \in S} D_{KL}(P_i \parallel Q) + \lambda \sum_{c=1}^C \max\left(0, \gamma - \sum_{i \in S} P_i(c)\right) \right) $$

- $\sum_{i \in S} D_{KL}(P_i \parallel Q)$: The cumulative divergence from the global ideal.
- $\lambda$: A massive scaling penalty coefficient (e.g., $10^6$).
- $\gamma$: The starvation threshold. If any class $c$ falls below this threshold, the heavy penalty forces the algorithm to reject that combination.

### 2.3 Algorithmic Fallback: Dynamic K-Scaling
If the mathematical penalty in FREQSEL still yields a candidate pool missing a critical class, we implement a programmatic Dynamic K-Scaling logic loop to guarantee class coverage without discarding optimized clients.

1. **Verify Coverage**: Let $U_K = \bigcup_{i \in K} \{c \mid P_i(c) > 0\}$ be the set of unique classes present in the initial pool $K$.
2. **Evaluate**: If $|U_K| < C$ (a class is missing), identify the missing class $c_{\text{miss}} \notin U_K$.
3. **Dynamic Expansion**: We dynamically expand the candidate pool size. Increment pool size $K_{\text{new}} = K + 1$.
4. **Selection & Addition**: Find the client $j \notin K$ in the excluded pool that possesses the maximum density of $c_{\text{miss}}$. Append client $j$ to the pool: $K_{\text{new}} \leftarrow K \cup \{j\}$.
5. **Termination**: Repeat $K_{\text{new}} \leftarrow K_{\text{new}} + 1$ until $|U_{K_{\text{new}}}| = C$.

### 2.4 BSFL Final Scheduling (UCB Formulation)
From the dynamically scaled candidate pool $K_{\text{new}}$, we select the final subset of $m$ clients for communication round $t$ using the Upper Confidence Bound (UCB) algorithm.

$$ A_t = \arg\max_{\substack{S \subset K_{\text{new}} \\ |S|=m}} \sum_{i \in S} \left( R_i + c \sqrt{\frac{\ln t}{N_i}} \right) $$

- $A_t$: The final selected set of $m$ clients.
- $R_i$: Historical empirical reward (exploitation), inversely proportional to latency.
- $c$: Exploration hyperparameter.
- $N_i$: Total times client $i$ has been selected prior to round $t$. The term $\sqrt{\frac{\ln t}{N_i}}$ forces exploration of rarely sampled clients.

### 2.5 Global Model Aggregation
The central server aggregates the locally trained ResNet-18 updates from the $m$ scheduled clients using Federated Averaging (FedAvg).

$$ w_{t+1} = \sum_{i \in A_t} \frac{n_i}{n_{A_t}} w_{t+1}^i \quad \text{where} \quad n_{A_t} = \sum_{i \in A_t} n_i $$

- $w_{t+1}$: The newly aggregated global model weights for round $t + 1$.
- $w_{t+1}^i$: The locally updated weights from client $i$.
- $\frac{n_i}{n_{A_t}}$: The weighting coefficient based on the number of local data samples $n_i$, ensuring statistically robust updates.

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
