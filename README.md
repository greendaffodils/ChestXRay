# 🩺 Graph-Augmented Medical Diagnosis from Chest X-rays

A research-oriented medical AI system that combines deep visual representation learning, disease knowledge graphs, and graph-based reasoning to perform robust and interpretable chest X-ray diagnosis.

# 📌 Overview

Chest X-ray interpretation is a core clinical task, but purely image-based deep learning models often ignore structured relationships between diseases (e.g., co-occurrence, clinical dependencies).

This project proposes a multi-stage pipeline that integrates:

CNN-based visual learning (DenseNet-121)

Patient-wise evaluation to prevent data leakage

Disease co-occurrence knowledge graphs

Graph Neural Networks (GNNs)

(Planned) Graph-RAG style reasoning and explanations

The result is a context-aware, interpretable, and research-grade medical AI system.

# 🎯 Key Contributions

✔ Patient-level train/validation/test splitting (no leakage)

✔ Strong CNN baseline trained on large-scale chest X-ray data

✔ Extraction of deep image embeddings (not just predictions)

✔ Construction of a disease knowledge graph from data

✔ Semantic node features using biomedical language models

✔ Foundation for CNN + GNN fusion and Graph-RAG reasoning



# 🧠 Methodology (High-Level)

Chest X-ray Image
        ↓
CNN (DenseNet-121)
        ↓
Image Embeddings (1024-D)
        ↓
Disease Probability Predictions
        ↓
Disease Knowledge Graph (Co-occurrence + Semantics)
        ↓
Graph Neural Network (GNN)
        ↓
Refined Predictions + Explainable Reasoning



# 📂 Dataset

NIH ChestX-ray14 Dataset

112,120 frontal chest X-ray images

14 thoracic disease labels

Multi-label classification setting

Real clinical data from NIH Clinical Center

# Labels
Atelectasis, Cardiomegaly, Effusion, Infiltration,
Mass, Nodule, Pneumonia, Pneumothorax,
Consolidation, Edema, Emphysema, Fibrosis,
Pleural Thickening, Hernia

Important Note

All dataset splits are performed patient-wise, ensuring that no patient appears in more than one split — a critical requirement for medical AI research.

⚙️ Project Structure
├── data/

│   ├── train_patient_split.csv

│   ├── val_patient_split.csv

│   └── test_patient_split.csv

│

├── models/

│   ├── densenet121.py

│   └── gnn_models.py

│

├── embeddings/

│   ├── train_embeddings.npy

│   ├── val_embeddings.npy

│   └── test_embeddings.npy

│

├── graph/

│   ├── nodes.csv

│   ├── edges.csv

│   └── node_embeddings.npy

│

├── notebooks/

│   ├── phase1_data_analysis.ipynb

│   ├── phase2_cnn_training.ipynb

│   └── phase3_graph_construction.ipynb

│

└── README.md



# 🔬 Phase-wise Breakdown

Phase 1 — Data Analysis & Preparation

Dataset exploration and cleaning

Multi-hot label encoding

Patient-wise splitting

Class imbalance analysis



Phase 2 — Deep Visual Modeling

DenseNet-121 (ImageNet-pretrained)

Multi-label classification with BCEWithLogitsLoss

Class imbalance handling via positive class weights

AUROC-based evaluation

Extraction of deep image embeddings



Phase 3 — Knowledge Graph Construction

Disease co-occurrence matrix from training data

Graph construction (nodes + weighted edges)

Semantic node features using BioBERT-based embeddings



Phase 4  — Graph Neural Networks

GraphSAGE / GAT over disease graph

Learning disease-level representations

Modeling label dependencies explicitly



Phase 5  — CNN + GNN Fusion

Fusion of image embeddings with graph embeddings

Improved prediction consistency and robustness



Phase 6 — Graph-RAG Reasoning

Retrieval of graph-based clinical context

Structured evidence packs for predictions

Natural-language explanations of model decisions





# 📊 Evaluation Protocol

Primary Metric: AUROC (per-class + mean)

Why AUROC: Robust to class imbalance, standard in medical imaging

No test-set exposure during training or validation



# 🧪 Interpretability (Planned)

Grad-CAM visual explanations for CNN predictions

Graph-based explanation of disease co-occurrence

Combined visual + relational interpretability



# 🚀 How to Run (High-Level)

Download dataset (via Kaggle)

Run notebook for preprocessing

Train CNN (Phase 2)

Extract embeddings

Build disease graph (Phase 3)

Train GNN and fusion models (Phase 4–5)



# 🎓 Research Motivation

Pure CNN-based systems treat diseases as independent labels.
This project explicitly models clinical relationships between diseases, bringing the system closer to how radiologists reason.

The design aligns with research directions seen in:

MICCAI

NeurIPS (Medical AI workshops)

Clinical decision-support systems

# 📚 References

Wang et al., ChestX-ray14: Hospital-scale Chest X-ray Database

Rajpurkar et al., CheXNet

Kipf & Welling, Graph Convolutional Networks

Hamilton et al., GraphSAGE


# 🤝 Acknowledgements

NIH Clinical Center, open-source ML community, and prior work in medical AI that inspired this pipeline.
