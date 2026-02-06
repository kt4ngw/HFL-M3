## Accelerating Hierarchical Federated Learning under Mobility via Model Migration in Cloud–Edge–End Collaborative Networks

The paper has been submitted by IEEE Trans. Mobile Comput. 

**Title:** Accelerating Hierarchical Federated Learning under Mobility via Model Migration in Cloud–Edge–End Collaborative Networks

**Author:**  Jian Tang, Xiaoyu Xia, Ibrahim Khalil, Mengsha Kou, Minghui Liwang, Jer Shyuan Ng, Xiuhua Li, and Xianbin Wang


### 1. Background
Deploying hierarchical federated learning (HFL) over cloud-edge-end collaborative networks has emerged as a prevalent paradigm for privacy-preserving model training.
Nevertheless, the non-independent and identically distributed (Non-IID) data characteristics across mobile clients (MCs) continue to present an inherent challenge that HFL alone cannot fully address.
This issue is further aggravated by MC mobility, which jointly hinders the stable convergence of the global model.
Most existing approaches tackle these problems primarily from the MC side, relying on implicit strategies such as model or data fusion, while overlooking the collaborative potential of edge servers (ESs).

### 2. Proposed HFL-M3 Framework
We propose an HFL under mobility via model migration (HFL-M3) framework to accelerate the global model convergence.
Prior to training, the framework organizes MCs into virtual groups with approximately balanced class distributions.
During training, it establishes logical mappings between each ES’s physical MC group and its corresponding target virtual group, and applies local adjustments exclusively to mismatched MCs.
This design promotes more class-balanced gradient aggregation at the ESs, thereby facilitating faster convergence of the global model.

### 3. Experiments
```
python main.py --server proposed
```
Note: The server is currently down; the code will be updated shortly.


