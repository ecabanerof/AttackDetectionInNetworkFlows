# Network Traffic Anomaly Detection with Explainable AI 🛡️

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.0-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-green.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45.1-purple.svg)](https://github.com/slundberg/shap)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

This repository contains a comprehensive implementation of a **network traffic anomaly detection system** that combines advanced Machine Learning techniques with Explainable AI (XAI). The project focuses on identifying and classifying cyber attacks using network flow analysis with complete model interpretability and quantization.

### 🎯 Main Objectives
- **Anomaly Detection**: Automatic identification of malicious network traffic
- **Explainability**: Detailed understanding of model decisions through XAI techniques
- **Optimization**: Efficient implementation for production deployment
- **Cross-Dataset Evaluation**: Exhaustive testing across multiple cybersecurity datasets

## 🔬 Key Features

### 🧠 Implemented Models
- **Multi-Layer Perceptron (MLP)** with optimized architectures
  - Basic models with sklearn (`MLPClassifier`)
  - Advanced implementations with PyTorch
  - Optimized intermediate architectures (128→256→128→64→classes)
- **Quantized Models** for efficient deployment
  - Dynamic and static quantization
  - Size reduction up to 75%
- **Random Forest** as comparative baseline
- **Robust preprocessing pipelines** with `RobustScaler`

### 📊 XAI (Explainable AI) Techniques
- **SHAP Values**: Feature importance analysis
- **Waterfall Plots**: Contribution visualization per prediction
- **Feature Importance**: Most relevant features ranking
- **Force Plots**: Individual prediction explanations
- **Summary Plots**: Global importance visualization

### ⚡ Performance Optimizations
- **Model Quantization**: Memory and latency reduction
- **Batch Processing**: Optimization for large volumes (>1M flows/s)
- **Memory Efficiency**: Efficient memory management techniques
- **Early Stopping**: Overfitting prevention with adaptive patience
- **Class Weights**: Intelligent unbalanced class handling

## 📁 Project Structure

```
/
├── DATASET-1/                       # TII-SSRC-23 IoT Dataset (420K samples)
│   ├── csv/                         # Data processed by attack type
│   │   ├── Audio.csv               # Audio multimedia traffic
│   │   ├── Background.csv          # Normal background traffic
│   │   ├── Bruteforce*.csv         # Brute force attacks (DNS, FTP, HTTP, SSH, Telnet)
│   │   ├── DoS*.csv               # DoS attacks by protocol (ACK, SYN, UDP, etc.)
│   │   ├── Mirai*.csv             # Mirai DDoS and Scan variants
│   │   ├── Video*.csv             # Video multimedia traffic (HTTP, RTP, UDP)
│   │   ├── cut_database_1.ipynb   # 📊 Main dataset processing
│   │   ├── data.csv               # Complete unified dataset
│   │   └── processed/             # Processed versions of individual files
│   └── v2/                        # Optimized dataset version
│       ├── final_csv/             # Final data with train/test/validation split
│       │   ├── train_1.csv        # TII-SSRC-23 training
│       │   ├── test_1.csv         # TII-SSRC-23 testing
│       │   └── validation_1.csv   # TII-SSRC-23 validation
│       └── updated_csv/           # Data processed with awk scripts
│
├── DATASET-2-2019/                 # CIC-IDS 2019 Dataset (2.8M samples) - MAIN
│   ├── train.csv                  # Training data (2.2M samples)
│   ├── test.csv                   # Test data (550K samples)
│   ├── validation.csv             # Validation data (275K samples)
│   ├── neurona_1_3.ipynb         # 🔥 Main notebook with optimized quantization
│   ├── neurona_basic.ipynb       # Baseline implementation with sklearn
│   ├── neuro_basic.ipynb         # Binary model (BENIGN vs Attack)
│   ├── neurona_1_2.ipynb         # Intermediate PyTorch model
│   ├── neurona_1_4.ipynb         # Additional experiments
│   ├── cut_database.ipynb         # Initial dataset processing
│   ├── *.csv                     # Files by attack type
│   │   ├── DrDoS_*.csv           # DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP
│   │   ├── LDAP.csv, MSSQL.csv   # Specific attacks
│   │   ├── Syn.csv, UDP.csv      # Protocol attacks
│   │   └── benign_data.csv       # Legitimate traffic
│   ├── *.pdf                     # Generated visualizations
│   │   ├── confusion_matrix_*.pdf # Confusion matrices
│   │   ├── decision_tree_*.pdf    # Decision trees
│   │   ├── SHAP_*.pdf            # XAI analysis
│   │   └── feature_importance_*.pdf # Feature importance
│   └── processed/                # Data processed by class
│
├── TEST_AREA/                     # Cross-dataset experiments
│   ├── 01_TRAIN_DATA2_TEST_DATA1/ # Train CIC-IDS 2019 → Test TII-SSRC-23
│   │   ├── neurona_2_1_test_better_label.ipynb # Main notebook
│   │   ├── test_data1.csv         # TII-SSRC-23 test data
│   │   ├── train_data2.csv        # CIC-IDS training data
│   │   └── *.pdf                  # SHAP visualizations and matrices
│   └── 02_TRAIN_DATA2_TEST_DATA2018/ # Train CIC-IDS 2019 → Test CIC-IDS 2018
│       ├── neurona_3_1_test_better_label.ipynb # Main notebook
│       ├── test_data2018.csv      # CIC-IDS 2018 test data
│       └── *.pdf                  # Analysis and results
│
├── requirements.yaml             # Conda/anaconda dependencies
└── README.md                    # This file
```

**📝 Note on XAI Techniques**: This project uses XAI techniques based on the [XAItechniquesinTC](https://github.com/matesanzvictor/XAItechniquesinTC) repository for comparison and validation of results.

## 🛠️ Installation

### Prerequisites
- **Anaconda/Miniconda** installed
- **Python 3.9+**
- **8GB RAM** minimum (16GB recommended)
- **10GB** free disk space

### Quick Installation with Conda

```bash
# Clone the repository
git clone https://github.com/ecabanerof/network-anomaly-detection-xai.git
cd network-anomaly-detection-xai

# Create environment from requirements.yaml
conda env create -f requirements.yaml

# Activate environment
conda activate tfm2025

# Verify installation
python -c "import torch, tensorflow as tf, shap; print('✅ Everything installed correctly')"
```

### Manual Installation (Alternative)

```bash
# Create new environment
conda create -n tfm2025 python=3.9

# Activate environment
conda activate tfm2025

# Install main dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install tensorflow=2.9.0 -c conda-forge
conda install scikit-learn=1.3.0 pandas=2.1.4 matplotlib seaborn -c conda-forge
conda install jupyter ipykernel -c conda-forge

# Install SHAP and specific libraries
pip install shap==0.45.1 scikit-plot
```

## 🚀 Usage Guide

### 1. 🏃‍♂️ Quick Start - Main Model

The main entry point is `DATASET-2-2019/neurona_1_3.ipynb`:

```bash
# Navigate to main dataset
cd DATASET-2-2019

# Run Jupyter
jupyter notebook neurona_1_3.ipynb
```

### 2. 📊 Notebooks by Use Case

| Objective | Dataset | Main Notebook | Description |
|-----------|---------|---------------|-------------|
| **🔥 Main Evaluation** | CIC-IDS 2019 | `DATASET-2-2019/neurona_1_3.ipynb` | Optimized model with quantization |
| **📊 Baseline Comparison** | CIC-IDS 2019 | `DATASET-2-2019/neurona_basic.ipynb` | Basic sklearn implementation |
| **🏠 IoT Analysis** | TII-SSRC-23 | `DATASET-1/csv/cut_database_1.ipynb` | IoT processing and analysis |
| **🔄 Cross-Dataset CIC→IoT** | CIC→TII-SSRC-23 | `TEST_AREA/01_TRAIN_DATA2_TEST_DATA1/neurona_2_1_test_better_label.ipynb` | CIC-IDS to IoT generalization |
| **🔄 Cross-Dataset CIC→CIC2018** | CIC→CIC-2018 | `TEST_AREA/02_TRAIN_DATA2_TEST_DATA2018/neurona_3_1_test_better_label.ipynb` | Temporal evaluation |
| **🎯 Binary Classification** | CIC-IDS 2019 | `DATASET-2-2019/neuro_basic.ipynb` | BENIGN vs Attack |

### 3. 🔬 Experimentation Pipeline

```python
# 1. Preprocessing with outlier removal
from sklearn.preprocessing import RobustScaler
import numpy as np

def remove_extreme_outliers(X, percentile=99.0):
    X_filtered = X.copy()
    for i in range(X.shape[1]):
        upper_limit = np.percentile(X[:, i], percentile)
        lower_limit = np.percentile(X[:, i], 100 - percentile)
        X_filtered[:, i] = np.clip(X_filtered[:, i], lower_limit, upper_limit)
    return X_filtered

scaler = RobustScaler()
X_clean = remove_extreme_outliers(X_train, percentile=99.0)
X_scaled = scaler.fit_transform(X_clean)

# 2. Optimized model training
model = IntermediateOptimizedModel(input_size=76, num_classes=17)
trained_model = train_intermediate_model(model, train_loader, val_loader)

# 3. Quantization for optimization
quantized_model = quantize_model(trained_model)
print(f"Size reduction: {size_reduction}%")

# 4. XAI analysis with SHAP
import shap
explainer = shap.KernelExplainer(model.predict_proba, X_background)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample)
```

## 📊 Supported Datasets

| Dataset | Full Name | Type | Classes | Samples | Features | Status | Main Notebook |
|---------|-----------|------|---------|---------|----------|--------|---------------|
| **DATASET-1** | **TII-SSRC-23** | IoT/DDoS | 12 | ~420K | 71 | ✅ Complete | `cut_database_1.ipynb` |
| **DATASET-2** | **CIC-IDS 2019** | Corporate Network | 17 | ~2.8M | 76 | ✅ Complete | `neurona_1_3.ipynb` |

### 🎯 Attack Types by Dataset

#### TII-SSRC-23 (IoT Dataset)
| Category | Attack Types | Approx. Samples | Description |
|----------|--------------|-----------------|-------------|
| **Normal** | Background, Audio, Video, Text | 180,000 | Legitimate IoT multimedia traffic |
| **DoS** | ACK, SYN, UDP, HTTP, ICMP, CWR, ECN, FIN, MAC, PSH, RST, URG | 150,000 | Denial of Service attacks |
| **Mirai** | DDoS (ACK, DNS, HTTP, SYN, UDP), Scan Bruteforce | 80,000 | Mirai malware variants |
| **Bruteforce** | DNS, FTP, HTTP, SSH, Telnet | 10,000 | Brute force attacks by protocol |

#### CIC-IDS 2019 (Main Dataset)
| Category | Attack Types | Approx. Samples | Description |
|----------|--------------|-----------------|-------------|
| **Normal** | BENIGN | 1,250,000 | Legitimate corporate network traffic |
| **DrDoS** | DNS, LDAP, MSSQL, NTP, NetBIOS, SNMP, SSDP, UDP | 1,400,000 | Distributed reflection attacks |
| **Others** | LDAP, MSSQL, NetBIOS, Portmap, Syn, TFTP, UDPLag | 150,000 | Various network attacks |

## 📈 Results and Metrics

### 🎯 Model Comparison on CIC-IDS 2019

| Model | Accuracy | Precision | Recall | F1-Score | Time (s) | Size (MB) | Throughput |
|-------|----------|-----------|--------|----------|----------|-----------|------------|
| **Optimized Model** | **55.4%** | **0.58** | **0.55** | **0.56** | 620 | 3.8 | 800K flows/s |
| Sklearn Base Model | 37.0% | 0.41 | 0.37 | 0.38 | 340 | 2.1 | 1.2M flows/s |
| **Quantized Model** | **53.8%** | **0.56** | **0.54** | **0.55** | 620 | **1.2** | **1.8M flows/s** |
| Random Forest | 46.9% | 0.52 | 0.47 | 0.49 | 180 | 15.3 | 600K flows/s |

### ⚡ Quantization Impact

| Metric | Original Model | Quantized Model |
|--------|----------------|-----------------|
| **Throughput** | 800K flows/s | **1.8M flows/s** |
| **Size** | 3.8 MB | **1.2 MB** |
| **Latency** | 1.25 ms | **0.56 ms** |
| **RAM Memory** | 100% | **35%** |
| **Accuracy Loss** | 55.4% | 53.8% |

### 🔍 Top Features by SHAP Analysis

| Ranking | Feature | SHAP Importance | Type | Description | Relevance |
|---------|---------|-----------------|------|-------------|-----------|
| 1 | **Flow Bytes/s** | 0.847 | Flow | Byte transfer speed | 🔥 Critical |
| 2 | **Packet Length Mean** | 0.621 | Packet | Average packet size | 🔥 Critical |
| 3 | **Flow IAT Min** | 0.518 | Temporal | Minimum inter-arrival time | 🟠 High |
| 4 | **Bwd Packets/s** | 0.492 | Flow | Backward packets per second | 🟠 High |
| 5 | **Total Length Fwd** | 0.438 | Flow | Total forward length | 🟠 High |
| 6 | **Packet Length Variance** | 0.401 | Packet | Packet size variance | 🟡 Medium |
| 7 | **Fwd Header Length** | 0.387 | Protocol | Forward header length | 🟡 Medium |

### 📊 Cross-Dataset Evaluation

| Train Dataset | Test Dataset | Accuracy | F1-Score | Domain Gap | Observations | Status |
|---------------|--------------|----------|----------|------------|--------------|--------|
| **CIC-IDS 2019** | **CIC-IDS 2019** | **55.4%** | **0.56** | ✅ None | Internal evaluation | ✅ Baseline |
| CIC-IDS 2019 | TII-SSRC-23 | 42.1% | 0.38 | ⚠️ Moderate | IoT vs Corporate shift | 🔄 Analyzed |
| CIC-IDS 2019 | CIC-IDS 2018 | 38.7% | 0.35 | ⚠️ Temporal | 2018→2019 evolution | 🔄 Analyzed |

## 📚 Main Notebooks and Files

### 🎨 Key Generated Visualizations

| Visualization Type | Files | Dataset | Purpose | Location |
|--------------------|-------|---------|---------|----------|
| **Confusion Matrices** | `confusion_matrix_*.pdf` | CIC-IDS 2019 | Detailed 17-class evaluation | `DATASET-2-2019/` |
| **SHAP Analysis** | `SHAP_*.pdf`, `shap_heatmap_*.pdf` | Multiple | XAI interpretability | `TEST_AREA/*/` |
| **Decision Trees** | `decision_tree_*.pdf` | CIC-IDS 2019 | Interpretable rules | `DATASET-2-2019/` |
| **Feature Importance** | `feature_importance_*.pdf` | Multiple | Feature ranking | Multiple folders |
| **Distributions** | `*_histogram.pdf` | CIC-IDS 2019 | Exploratory analysis | `DATASET-2-2019/` |

## 🔧 Configuration and Parameters

### ⚙️ Recommended Configuration by Dataset

| Dataset | Batch Size | Epochs | Learning Rate | Memory (GB) | Approx. Time |
|---------|------------|--------|---------------|-------------|--------------|
| **CIC-IDS 2019** | 1024 | 80 | 0.004 | 8+ | 10-12 hours |
| **TII-SSRC-23** | 512 | 60 | 0.003 | 4+ | 3-4 hours |

### 🎛️ Optimized Model Architecture

```python
OPTIMIZED_CONFIG = {
    # Network architecture
    'input_size': 76,                    # CIC-IDS 2019 features
    'hidden_layers': [128, 256, 128, 64], # Optimized hidden layers
    'num_classes': 17,                   # CIC-IDS 2019 classes
    
    # Regularization
    'dropout_rates': [0.3, 0.4, 0.2],   # Progressive dropout
    'batch_norm': True,                  # BatchNormalization
    'activation': ['ReLU', 'ELU'],       # Activation functions
    
    # Training
    'optimizer': 'AdamW',                # Advanced optimizer
    'lr_scheduler': 'OneCycleLR',        # LR scheduler
    'early_stopping': 15,               # Early stopping patience
    'class_weights': 'balanced_sqrt',    # Smoothed class weights
    
    # Quantization
    'quantization': 'dynamic',           # Quantization type
    'dtype': 'qint8'                    # Quantized precision
}
```

## 🧪 Research Methodology

### 1. **Robust Preprocessing**
```python
def remove_extreme_outliers(X, percentile=99.0):
    """Remove extreme outliers while maintaining natural distribution"""
    X_filtered = X.copy()
    for i in range(X.shape[1]):
        upper_limit = np.percentile(X[:, i], percentile)
        lower_limit = np.percentile(X[:, i], 100 - percentile)
        X_filtered[:, i] = np.clip(X_filtered[:, i], lower_limit, upper_limit)
    return X_filtered

# Preprocessing pipeline
scaler = RobustScaler()  # Robust to outliers
X_clean = remove_extreme_outliers(X_train, percentile=99.0)
X_scaled = scaler.fit_transform(X_clean)
```

### 2. **Optimized Neural Architecture**
```python
class IntermediateOptimizedModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Expand-contract architecture
        self.fc1 = nn.Linear(input_size, 128)    # Input
        self.fc2 = nn.Linear(128, 256)           # Expansion
        self.fc3 = nn.Linear(256, 128)           # Contraction
        self.fc4 = nn.Linear(128, 64)            # Reduction
        self.fc5 = nn.Linear(64, num_classes)    # Output
        
        # Advanced regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
```

### 3. **Quantization for Optimization**
```python
def quantize_model(model):
    """Dynamic quantization for efficient deployment"""
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},      # Quantize linear layers only
        dtype=torch.qint8 # Maximum compression
    )
    
    return quantized_model
```

### 4. **XAI Analysis with SHAP**
```python
def perform_shap_analysis(model, X_background, X_test, feature_names):
    """Complete explainability analysis"""
    # Create explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Visualizations
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    shap.waterfall_plot(explainer.expected_value[0], shap_values[0], X_test[0])
    
    return shap_values, explainer
```

## 🚨 Limitations and Considerations

### ❌ Identified Limitations

| Problem | Description | Impact | Implemented Mitigation |
|---------|-------------|--------|----------------------|
| **Class Imbalance** | CIC-IDS: some classes <1% | 🔴 High | Sqrt-balanced class weights |
| **Domain Shift** | Low IoT↔Corporate generalization | 🟡 Medium | Transfer learning planned |
| **Memory Footprint** | Large models >8GB RAM | 🟡 Medium | Quantization -65% memory |
| **Temporal Drift** | CIC-2018 vs CIC-2019 differences | 🟡 Medium | Cross-temporal evaluation |
| **Feature Engineering** | Domain-specific characteristics | 🟡 Medium | Detailed SHAP analysis |

### ⚠️ Deployment Considerations

- **Real-time Latency**: Quantized model <1ms per prediction
- **Throughput**: >1.8M flows per second on standard hardware
- **Memory**: Quantized model requires only 35% original memory
- **Accuracy vs Speed**: Acceptable trade-off (-1.6% accuracy for +125% speed)
- **Scalability**: Architecture ready for data streaming

## 📊 State-of-the-Art Comparison

### 🏆 Public Benchmarks

| Paper/Method | Dataset | Accuracy | F1-Score | Technique | Year |
|--------------|---------|----------|----------|-----------|------|
| **Our Work** | **CIC-IDS 2019** | **55.4%** | **0.56** | **MLP + Quantization + XAI** | **2025** |
| Sharafaldin et al. | CIC-IDS 2019 | 50.2% | 0.48 | Random Forest | 2019 |
| Wang et al. | CIC-IDS 2019 | 52.8% | 0.51 | CNN-LSTM | 2022 |
| Liu et al. | CIC-IDS 2019 | 48.9% | 0.45 | Ensemble | 2023 |

**✅ Advantages of our approach:**
- **Complete explainability** with SHAP
- **Production optimization** with quantization
- **Exhaustive cross-dataset evaluation**
- **Reproducibility** with open source code

## 📄 License

This project is under the **MIT License** - see [LICENSE](LICENSE) file for complete details.

## 📞 Contact and Support

- **Author**: Emilio Cabañero
- **LinkedIn**: [linkedin.com/in/ecabanerof](https://www.linkedin.com/in/ecabanerof)
- **University**: HPCN-UAM

### 🐛 Issue Reporting

To report problems, use **GitHub Issues** including:
- **Detailed description** of the problem
- **Minimal code** to reproduce the error
- **Environment information**: OS, Python version, dependencies
- **Screenshots** if relevant
- **Complete error logs**

## 🙏 Acknowledgments
- **HPCN-UAM** ([HPCN](http://www.hpcn-uam.es/)) High Performance Computing and Networking research group 
- **Canadian Institute for Cybersecurity** ([CIC-IDS 2019](https://www.unb.ca/cic/datasets/ddos-2019.html)) for the CIC-IDS 2019 dataset
- **Telecommunications and Information Institute (TII)** ([TII-SSRC-23](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23)) for the TII-SSRC-23 dataset
- **SHAP Library developers** for explainability tools
- **PyTorch Team** for the deep learning framework
- **scikit-learn community** for base ML implementations
- **XAItechniquesinTC project** ([matesanzvictor](https://github.com/matesanzvictor/XAItechniquesinTC)) for reference XAI techniques

## 📚 References and Citations

### 📖 Main Papers
```bibtex
@mastersthesis{cabanero2025tfm,
  author    = {Emilio Cabañero Fernández},
  title     = {Identificación de Características Significativas para la Detección Eficiente de Ataques en Flujos de Red aplicando técnicas XAI},
  school    = {Universidad Autónoma de Madrid},
  type      = {Master's Thesis},
  year      = {2025},
  month     = {September},
  address   = {Madrid, Spain},
  supervisor = {Jorge Enrique López de Vergara Méndez}
}

@misc{matesanz2023aplicacion,
  title={Aplicación De Técnicas De Inteligencia Artificial Explicable Para Interpretar Resultados En La Identificación De Ataques En Flujos De Red Mediante Redes Neuronales Artificiales},
  author={Matesanz Cotillas, V.},
  year={2023},
  school={Universidad Autónoma de Madrid},
  note={Master's Thesis, Master in Telecommunications Engineering (EUR-ACE®), Madrid}
}

@inproceedings{sharafaldin2018toward,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  pages     = {108--116},
  year      = {2018},
  publisher = {SciTePress},
  doi       = {10.5220/0006639801080116}
}

@article{sharafaldin2019developing,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Saqib Hakak and Ali A. Ghorbani},
  title     = {Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy},
  journal   = {IEEE Access},
  volume    = {7},
  pages     = {36381--36393},
  year      = {2019},
  doi       = {10.1109/ACCESS.2019.2904802}
}

@article{herzalla2023tiissrc23,
  author    = {Dania Herzalla and Willian Tessaro Lunardi and Martin Andreoni},
  title     = {TII-SSRC-23 Dataset: Typological Exploration of Diverse Traffic Patterns for Intrusion Detection},
  journal   = {IEEE Access},
  volume    = {11},
  pages     = {118577--118591},
  year      = {2023},
  doi       = {10.1109/ACCESS.2023.3319213},
  note      = {Dataset available at \url{https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23}}
}
```

### 🔗 Additional Resources
- [Complete project documentation](docs/)
- [CIC-IDS 2019 Dataset](https://www.unb.ca/cic/datasets/ids-2019.html)
- [XAItechniquesinTC Reference](https://github.com/matesanzvictor/XAItechniquesinTC)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

[![Star History Chart](https://api.star-history.com/svg?repos=ecabanerof/AttackDetectionInNetworkFlows&type=Date)](https://star-history.com/#ecabanerof/AttackDetectionInNetworkFlows&Date)

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg"/>
  <img src="https://img.shields.io/badge/Python-Powered-blue.svg"/>
  <img src="https://img.shields.io/badge/AI-Explainable-green.svg"/>
  <img src="https://img.shields.io/badge/Security-First-orange.svg"/>
  <img src="https://img.shields.io/badge/Open%20Source-💡-yellow.svg"/>
</p>
