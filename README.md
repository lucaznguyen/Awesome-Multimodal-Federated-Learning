# Awesome Multimodal Federated Learning (MMFL)

A curated reading list of **Multimodal Federated Learning (MMFL)** papers and datasets.  

**Scope:** papers where **federated learning is central** and the learning problem/model/data is **multimodal**.  

**Last updated:** 2025-12-29

> ⚠️ "All papers"" is a moving target. This list aims to be **as complete as possible** up to the date above. If anything is missing, please open an issue / PR!

---

## Contents

- [Papers](#papers)
  - [Surveys & Tutorials](#surveys--tutorials)
  - [Benchmarks, Toolkits & Libraries](#benchmarks-toolkits--libraries)
  - [Core MMFL Methods](#core-mmfl-methods)
  - [Missing Modality & Modality Incompleteness](#missing-modality--modality-incompleteness)
  - [Vision-Language & Foundation Models in FL](#vision-language--foundation-models-in-fl)
  - [Cross-Modal Retrieval / Hashing in FL](#cross-modal-retrieval--hashing-in-fl)
  - [Applications](#applications)
    - [Healthcare & Medical](#healthcare--medical)
    - [Human Activity Recognition, Wearables & IoT](#human-activity-recognition-wearables--iot)
    - [Autonomous Driving & Smart Transportation](#autonomous-driving--smart-transportation)
    - [Remote Sensing](#remote-sensing)
    - [Affective Computing: Emotion & Sentiment](#affective-computing-emotion--sentiment)
    - [Systems, Networking & Efficiency](#systems-networking--efficiency)
  - [Security, Privacy & Robustness](#security-privacy--robustness)
- [Datasets & Benchmarks](#datasets--benchmarks)
  - [Federated Multimodal Benchmarks](#federated-multimodal-benchmarks)
  - [Common Multimodal Datasets Used in MMFL](#common-multimodal-datasets-used-in-mmfl)
- [How to Contribute](#how-to-contribute)

---

## Papers

### Surveys & Tutorials

#### 2025
- **Multimodal Federated Learning: A Survey through the Lens of Different FL Paradigms** — Peng *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.21792)

- **A Survey on Vision-Language Models for Multimodal Federated Learning Tasks** — Yang (TechRxiv, 2025)  
  [Paper](https://www.techrxiv.org/users/956810/articles/1325916-a-survey-on-vision-language-models-for-multimodal-federated-learning-tasks)

#### 2024
- **A Survey of Multimodal Federated Learning: Background, Applications, and Perspectives** — Pan *et al.* (Multimedia Systems, 2024)  
  [Publisher](https://link.springer.com/article/10.1007/s00530-024-01386-z)

- **Advances and Applications of Multimodal Federated Learning: A Survey** — Barry *et al.* (2024)  
  [Paper](https://arxiv.org/abs/2404.14575)

#### 2023
- **Multimodal Federated Learning: A Survey** — Che *et al.* (Sensors, 2023)  
  [Paper](https://www.mdpi.com/1424-8220/23/15/6986)

- **Federated Learning on Multimodal Data: A Comprehensive Survey** — Lin *et al.* (Springer, 2023)  
  [Publisher](https://link.springer.com/chapter/10.1007/978-3-031-16086-8_2)

- **Multimodal Federated Learning in Healthcare: A Review** — Thrasher *et al.* (arXiv, 2023)  
  [Paper](https://arxiv.org/abs/2310.09650)

---

### Benchmarks, Toolkits & Libraries

#### 2025
- **Federated Continual Instruction Tuning (FCIT)** — Guo *et al.* (ICCV, 2025)  
  [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html) · [Code/Dataset](https://github.com/Ghy0501/FCIT)

#### 2024
- **FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data** — Xu *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2411.14717) · [Code](https://github.com/1xbq1/FedMLLM)

#### 2023
- **FedMultimodal: A Benchmark for Multimodal Federated Learning** — Feng *et al.* (KDD, 2023)  
  [Paper](https://arxiv.org/abs/2306.09486) · [Code](https://github.com/usc-sail/fed-multimodal)

---

### Core MMFL Methods

#### 2025
- **BlendFL: Blended Federated Learning for Handling Multimodal Data Heterogeneity** — (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2510.13266)

- **MASA: Multimodal Federated Learning Through Modality-Aware Structure Adaptation** — Guo *et al.* (IEEE TMM, 2025)  
  [Publisher](https://ieeexplore.ieee.org/document/10916948)

- **Federated Multimodal Learning with Dual Adapters and Selective Pruning for Communication and Computational Efficiency** — Nguyen *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2503.07552)

- **AproMFL: Adaptive Prototype-based Multimodal Federated Learning for Mixed Modalities and Heterogeneous Tasks** — (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2502.04400)

- **Quantum Federated Learning for Multimodal Data: A Modality-Agnostic Approach** — Pokharel *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2507.08217)

#### 2024
- **Adaptive Hyper-graph Aggregation for Modality-Agnostic Federated Learning** — Qi & Li (CVPR, 2024)  
  [Paper (PDF)](https://openaccess.thecvf.com/content/CVPR2024/papers/Qi_Adaptive_Hyper-graph_Aggregation_for_Modality-Agnostic_Federated_Learning_CVPR_2024_paper.pdf) · [Code](https://github.com/MM-Fed/HAMFL)

- **On Disentanglement of Asymmetrical Knowledge Transfer for Modality-Task Agnostic Federated Learning** — Chen & Zhang (AAAI, 2024)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28706)

- **Towards Multi-modal Transformers in Federated Learning (FedCola)** — Sun *et al.* (ECCV, 2024)  
  [Paper](https://arxiv.org/abs/2404.12467) · [Code](https://github.com/imguangyu/FedCola)

- **Resource-Efficient Federated Multimodal Learning via Layer-Wise Aggregation (LW-FedMML)** — Guo *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2407.15426)

- **Prioritizing Modalities: Flexible Importance Scheduling in Federated Multimodal Learning** — Dong *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2408.06549)

- **Communication-Efficient Multimodal Federated Learning: Joint Modality and Client Selection** — Yuan *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2401.16685)

- **Multimodal Federated Learning with Model Personalization** — Rahman & Nguyen (NeurIPS OPT Workshop, 2024)  
  [Paper](https://openreview.net/forum?id=y5Y0faJYH8)

#### 2023
- **Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training** — Zheng *et al.* (MobiSys, 2023)  
  [Paper](https://dl.acm.org/doi/10.1145/3581791.3596854) · [Code](https://github.com/xmouyang/Harmony
)

- **FedSea: Federated Learning via Selective Feature Alignment for Non-IID Multimodal Data** — Lyu *et al.* (IEEE TMM, 2023)  
  [Paper](https://ieeexplore.ieee.org/document/10185550)

- **CreamFL: Multimodal Federated Learning via Contrastive Representation Ensemble and Aggregation** — Yu *et al.* (ICLR, 2023)  
  [Paper](https://arxiv.org/abs/2302.08888) · [Code](https://github.com/FLAIR-THU/CreamFL)

- **FedMFS: Federated Multimodal Fusion Learning with Selective Modality Communication** — Feng *et al.* (arXiv, 2023)  
  [Paper](https://arxiv.org/abs/2302.13550)

- **FedMEKT: Distillation-based Embedding Knowledge Transfer for Multimodal Federated Learning** — Zhao *et al.* (arXiv, 2023)  
  [Paper](https://arxiv.org/abs/2304.08092)

#### 2022
- **FedMSplit: Correlation-Adaptive Federated Multi-Task Learning across Multimodal Split Networks** — Wang *et al.* (KDD, 2022)  
  [Paper](https://dl.acm.org/doi/10.1145/3534678.3539196)

- **A Unified Framework for Multi-Modal Federated Learning** — (Neurocomputing, 2022)  
  [Publisher](https://www.sciencedirect.com/science/article/pii/S0925231222006045)

- **Multimodal Federated Learning on IoT Data** — Zhao *et al.* (IoTDI, 2022)  
  [Paper](https://dl.acm.org/doi/10.1145/3498361.3538920)

- **Towards Optimal Multi-Modal Federated Learning on Non-IID Data with Hierarchical Gradient Blending** — Chen & Li (INFOCOM, 2022)  
  [Paper](https://ieeexplore.ieee.org/document/9796910)

- **Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning** — Yang *et al.* (AAAI, 2022)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20213)

#### 2021
- **FedCMR: Federated Cross-Modal Retrieval** — Zong *et al.* (SIGIR, 2021)  
  [Paper](https://dl.acm.org/doi/10.1145/3404835.3462989)

#### 2020
- **Federated Learning for Vision-and-Language Grounding Problems** — Liu *et al.* (AAAI, 2020)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6824)

#### 2019
- **Think Locally, Act Globally: Federated Learning with Local and Global Representations** — Liang *et al.* (NeurIPS FL Workshop, 2019)  
  [Paper](https://arxiv.org/abs/2001.01523)

---

### Missing Modality & Modality Incompleteness

#### 2025
- **Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data** — Nguyen *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2510.22880) · [Code](https://github.com/nmduonggg/PEPSY)

- **Multimodal Federated Learning with Missing Modalities through Feature Imputation Network** — Poudel *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.20232) · [Code](https://github.com/bhattarailab/FedFeatGen)

- **Multimodal Online Federated Learning with Modality Missing in Internet of Things** — Wang *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.16138)

- **MMiC: Mitigating Modality Incompleteness in Clustered Multimodal Federated Learning** — (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.06911) · [Code](https://github.com/gotobcn8/MMiC)

- **Leveraging Foundation Models for Multi-modal Federated Learning with Incomplete Modality (FedMVP)** — Che *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2406.11048) · [Code](https://github.com/mainaksingha01/FedMVP)

#### 2024
- **Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality (MFCPL)** — Le *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2401.13898)

- **Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection** — Saha *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2402.05294)

- **Robust Multimodal Federated Learning for Incomplete Modalities** — Chai *et al.* (Neural Networks, 2024)  
  [Publisher](https://www.sciencedirect.com/science/article/pii/S0893608024002567)

#### 2023
- **Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast (PmcmFL)** — Bao *et al.* (arXiv, 2023)  
  [Paper](https://arxiv.org/abs/2312.13508) · [Code](https://github.com/BaoGuangYin/PmcmFL)

---

### Vision-Language & Foundation Models in FL

#### 2025
- **Pilot: Building the Federated Multimodal Instruction Tuning Framework (FedMIT)** — Xiong *et al.* (AAAI, 2025)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/35476) · [arXiv](https://arxiv.org/abs/2501.13985)

- **Federated Continual Instruction Tuning (FCIT)** — Guo *et al.* (ICCV, 2025)  
  [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html) · [Code/Dataset](https://github.com/Ghy0501/FCIT)

- **FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models** — Singha *et al.* (ICCV, 2025)  
  [Paper](https://arxiv.org/abs/2504.20860)

- **Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data** — Phung *et al.* (ICCV, 2025)  
  [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Phung_Federated_Prompt-Tuning_with_Heterogeneous_and_Incomplete_Multimodal_Client_Data_ICCV_2025_paper.html)

- **FedVLP: Visual-aware Latent Prompt Generation for Federated Vision-Language Models** — Pan *et al.* (2025)  
  [Publisher](https://www.sciencedirect.com/science/article/pii/S1077314225001651)

- **Multi-Modal One-Shot Federated Ensemble Learning for Medical Data with Vision Large Language Model (FedMME)** — Wang *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2501.03292)

- **FedAPT: Federated Adversarial Prompt Tuning for Vision-Language Models** — (ACM, 2025)  
  [Publisher](https://dl.acm.org/doi/10.1145/3746027.3755387)

- **Federated CLIP for Resource-Efficient Heterogeneous Medical Applications** — (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2511.07929)

#### 2024
- **FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data** — Xu *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2411.14717) · [Code](https://github.com/1xbq1/FedMLLM)

- **Federated Text-driven Prompt Generation for Vision-Language Models** — Qiu *et al.* (ICLR, 2024)  
  [Paper](https://openreview.net/forum?id=NW31gAylIm) · [Code](https://github.com/boschresearch/FedTPG)

- **Global and Local Prompts Cooperation via Optimal Transport for Federated Learning** — (CVPR, 2024)  
  [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Global_and_Local_Prompts_Cooperation_via_Optimal_Transport_for_Federated_Learning_CVPR_2024_paper.html) · [Code](https://github.com/HongxiaLee/FedOTP)

- **FedDAT: Foundation Model Finetuning in Multimodal Heterogeneous Federated Learning** — (AAAI, 2024)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28772) · [Code](https://github.com/HaokunChen245/FedDAT)

- **Towards Multi-modal Transformers in Federated Learning (FedCola)** — Sun *et al.* (ECCV, 2024)  
  [Paper](https://arxiv.org/abs/2404.12467) · [Code](https://github.com/imguangyu/FedCola)

- **Leveraging Foundation Models for Multi-modal Federated Learning with Incomplete Modality (FedMVP)** — Che *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2406.11048)

#### 2023
- **pFedPrompt: Learning Personalized Prompt for Vision-Language Models in Federated Learning** — (WWW, 2023)  
  [Paper](https://dl.acm.org/doi/10.1145/3543507.3583518)

- **FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning** — Lu *et al.* (2023)  
  [Paper](https://arxiv.org/abs/2302.13485)

---

### Cross-Modal Retrieval / Hashing in FL

#### 2023
- **Prototype-guided Knowledge Transfer for Federated Unsupervised Cross-modal Hashing (PT-FUCH)** — Li *et al.* (ACM MM, 2023)  
  [Paper](https://dl.acm.org/doi/10.1145/3581783.3613837)

#### 2021
- **FedCMR: Federated Cross-Modal Retrieval** — Zong *et al.* (SIGIR, 2021)  
  [Paper](https://dl.acm.org/doi/10.1145/3404835.3462989) · [Code](https://github.com/hasakiXie123/FedCMR)

---

### Applications

#### Healthcare & Medical

##### 2025
- **FedMRG: Federated Medical Report Generation via Text-Image Collaborative Learning** — Metmer *et al.* (Multimedia Systems, 2025)  
  [Publisher](https://link.springer.com/article/10.1007/s00530-025-01725-5)

- **Multimodal Federated Learning with Missing Modalities through Feature Imputation Network** — Poudel *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.20232)

- **Multi-Modal One-Shot Federated Ensemble Learning for Medical Data with Vision Large Language Model (FedMME)** — Wang *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2501.03292)

##### 2024
- **Medical Report Generation based on Multimodal Federated Learning** — Chen *et al.* (Computerized Medical Imaging and Graphics, 2024)  
  [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0895611124000193)

- **Federated Modality-Specific Encoders and Multimodal Anchors for Personalized Brain Tumor Segmentation** — Ye *et al.* (AAAI, 2024)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28642)

- **A Federated Learning System with Data Fusion for Healthcare using Multi-Party Computation and Additive Secret Sharing** — Zhang *et al.* (Computer Communications, 2024)  
  [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0140366424000505)

- **Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection** — Saha *et al.* (arXiv, 2024)  
  [Paper](https://arxiv.org/abs/2402.05294)

##### 2021
- **Multimodal Melanoma Detection with Federated Learning** — Agbley *et al.* (2021)  
  [Paper](https://doi.org/10.1109/WAMTIP52925.2021.9589976)

---

#### Human Activity Recognition, Wearables & IoT

##### 2025
- **Multimodal Online Federated Learning with Modality Missing in Internet of Things** — Wang *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2505.16138)

##### 2024
- **Cross-Modal Federated Human Activity Recognition** — Yang *et al.* (IEEE TPAMI, 2024)  
  [Publisher](https://ieeexplore.ieee.org/document/10434084)

##### 2023
- **FL-FD: Federated Learning-based Fall Detection with Multimodal Data Fusion** — Panwar *et al.* (Information Fusion, 2023)  
  [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S1566253523002406)

##### 2022
- **Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning** — Yang *et al.* (AAAI, 2022)  
  [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20213)

- **Multimodal Federated Learning on IoT Data** — Zhao *et al.* (IoTDI, 2022)  
  [Paper](https://dl.acm.org/doi/10.1145/3498361.3538920)

---

#### Autonomous Driving & Smart Transportation

##### 2025
- **RoadFed: A Multimodal Federated Learning System for Road Hazard Detection** — (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2502.09978)

- **FedMultiEmo: Real-Time Emotion Recognition via Multimodal Federated Learning (In-Vehicle)** — Gül *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2507.15470)

##### 2024
- **FedCMD: Federated Cross-Modal Knowledge Distillation for Drivers Emotion Recognition** — (ACM TIST, 2024)  
  [Publisher](https://dl.acm.org/doi/10.1145/3643815)

- **FedUSL: A Federated Annotation Method for Driving Fatigue Detection based on Multimodal Sensing Data** — (ACM TOSN, 2024)  
  [Publisher](https://dl.acm.org/doi/10.1145/3631461)

##### 2023
- **AutoFed: Heterogeneity-Aware Federated Multimodal Learning for Robust Autonomous Driving** — Zheng *et al.* (MobiCom, 2023)  
  [Paper](https://arxiv.org/abs/2302.08646)

---

#### Remote Sensing

##### 2025
- **A Multimodal Federated Learning Framework for Remote Sensing Image Classification** — Büyüktaş *et al.* (IEEE TGRS / arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2503.10262)

##### 2023
- **Fedfusion: A Multimodal Federated Learning Framework for Remote Sensing Data Fusion** — Guo *et al.* (IEEE GRSL, 2023)  
  [Publisher](https://ieeexplore.ieee.org/document/10080932)

---

#### Affective Computing: Emotion & Sentiment

##### 2025
- **Federated Dialogue-Semantic Diffusion for Emotion Recognition in Conversations** — (NeurIPS, 2025)  
  [Paper](https://neurips.cc/virtual/2025/poster/116219)

- **FedSER-XAI: PSO-optimized Multi-stream Cross-attention Explainable Federated Learning for Speech Emotion Recognition** — Alkhamali *et al.* (Scientific Reports, 2025)  
  [Publisher](https://www.nature.com/articles/s41598-025-28686-z)

- **A Financial Multimodal Sentiment Analysis Model Based on Federated Learning (Text + Voice)** — (Preprint, 2025)  
  [Paper](https://www.preprints.org/manuscript/202506.0968)

##### 2024
- **Federated Learning for Multimodal Sentiment Analysis: Advancing Global Models with an Enhanced LinkNet Architecture** — (IEEE Access, 2024)  
  [Publisher](https://doi.org/10.1109/ACCESS.2024.3503290)

- **AFLEMP: Attention-based Federated Learning for Emotion Recognition using Multi-modal Physiological Data** — Gahlan *et al.* (Biomedical Signal Processing and Control, 2024)  
  [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S1746809424004117)

- **Federated Learning Inspired Privacy Sensitive Emotion Recognition System (F-MERS)** — Gahlan *et al.* (Cluster Computing, 2024)  
  [Publisher](https://link.springer.com/article/10.1007/s10586-023-04133-4)

- **Enhancing Emotion Recognition through Federated Learning: A Multimodal Approach** — Simić *et al.* (Applied Sciences, 2024)  
  [Paper](https://www.mdpi.com/2076-3417/14/4/1325)

##### 2023
- **Federated Meta-Learning for Emotion and Sentiment Aware Multi-modal Complaint Identification** — (EMNLP, 2023)  
  [Paper](https://aclanthology.org/2023.emnlp-main.734/)

---

#### Systems, Networking & Efficiency

##### 2025
- **One-shot Multimodal Federated Learning via Diverse Synthetic Feature Optimization (DSFO)** — Qi *et al.* (ACM Multimedia Asia, 2025)  
  [Publisher](https://dl.acm.org/doi/10.1145/3743093.3770969)

- **Latency-aware Multimodal Federated Learning over UAV-assisted Networks** — Shaon *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2510.01717)

- **Enhanced Distributed Multimodal Federated Learning (E-DMFL)** — Aga *et al.* (Electronics, 2025)  
  [Paper](https://www.mdpi.com/2079-9292/14/20/4024)

---

### Security, Privacy & Robustness

#### 2025
- **BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models** — Zhang *et al.* (arXiv, 2025)  
  [Paper](https://arxiv.org/abs/2508.08040)

#### 2023
- **Multi-Modal Vertical Federated Learning Framework Based on Homomorphic Encryption** — Gong *et al.* (2023)  
  [Paper](https://arxiv.org/abs/2303.12307)

---

## Datasets & Benchmarks

### Federated Multimodal Benchmarks
- **FedMultimodal** (KDD 2023): end-to-end benchmark pipeline + federated partitions across multiple multimodal tasks/datasets  
  Repo: https://github.com/usc-sail/fed-multimodal

- **FedMLLM** (2024): benchmark for federated fine-tuning of multimodal LLMs under modal heterogeneity  
  Repo: https://github.com/1xbq1/FedMLLM

- **FCIT** (ICCV 2025): federated continual instruction tuning benchmark (multiple instruction datasets)  
  Repo: https://github.com/Ghy0501/FCIT

---

### Common Multimodal Datasets Used in MMFL

> Many MMFL papers use *standard multimodal datasets* and create federated splits (client partitions, modality-missing patterns, non-IID label skew, etc.). Below are widely used datasets in MMFL literature.

#### Vision-Language / Image-Text
- MSCOCO Captions — https://cocodataset.org/
- Flickr30k — https://shannon.cs.illinois.edu/DenotationGraph/
- VQA v2 — https://visualqa.org/
- Hateful Memes — https://ai.facebook.com/datasets/hateful-memes/
- Conceptual Captions — https://ai.google.com/research/ConceptualCaptions/

#### Medical Vision-Language (Images + Reports)
- MIMIC-CXR — https://physionet.org/content/mimic-cxr/
- Open-I / IU X-Ray — https://openi.nlm.nih.gov/

#### Video / Audio / Text (Sentiment, Emotion, Conversation)
- CMU-MOSI — https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- CMU-MOSEI — https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
- MELD — https://affective-meld.github.io/
- IEMOCAP — https://sail.usc.edu/iemocap/

#### Audio-Visual Event / Action Recognition
- AVE — https://github.com/YapengTian/AVE_Dataset
- UCF-101 — https://www.crcv.ucf.edu/data/UCF101.php
- Kinetics-400 — https://deepmind.com/research/open-source/kinetics

#### 3D / Multi-view / Multi-sensor
- ModelNet40 — https://modelnet.cs.princeton.edu/
- MMAct (multimodal action dataset) — https://mmact19.github.io/

#### Social / Crisis / Multimedia
- CrisisMMD — https://crisisnlp.qcri.org/crisismmd

#### Autonomous Driving (Sensor Fusion)
- nuScenes — https://www.nuscenes.org/
- KITTI — https://www.cvlibs.net/datasets/kitti/
- Waymo Open Dataset — https://waymo.com/open/

#### Remote Sensing (Optical + SAR / Multi-sensor)
- SEN12MS — https://mediatum.ub.tum.de/1474000
- SpaceNet — https://spacenet.ai/

---

## How to Contribute

PRs are welcome! Please keep entries:
- **MMFL-focused** (federated + multimodal is essential, not incidental).
- Include **year**, **venue** (or “arXiv”), and at least a **paper link**.
- Prefer **official** PDF / arXiv and **official code** links.

**Paper entry format (recommended):**
- **Title** — Authors. (Venue, Year) [Paper](...) · [Code](...) · [Dataset](...)
