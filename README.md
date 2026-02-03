# FreDN: Frequency Domain Decomposition Network for Long-term Time Series Forecasting

Official implementation for AAAI 2026 paper.

## Overview

FreDN is a novel deep learning model for long-term time series forecasting that leverages frequency domain decomposition and dual-branch prediction for seasonal and trend components.

## Requirements

### Environment
- Python 3.9+
- PyTorch 2.7.1 (CUDA 12.6)
- CUDA-compatible GPU (optional, CPU mode supported)

### Installation

```bash
# Create conda environment
conda create -n fredn python=3.9
conda activate fredn

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The code supports the following datasets:
- **ETT** (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1, ETTm2
- **Electricity**: ECL dataset
- **Traffic**: Traffic dataset
- **Weather**: Weather dataset

Place datasets in `./dataset/` directory:
```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
└── weather/
    └── weather.csv
```

## Quick Start

### Training

Run experiments using provided scripts:

```bash
# ETTh1 dataset
bash scripts/ETTh1.sh

# Electricity dataset
bash scripts/ECL.sh

# Traffic dataset
bash scripts/traffic.sh

# Weather dataset
bash scripts/weather.sh
```

## Citation

If you find this work helpful, please cite:

```bibtex
@misc{an2025frednspectraldisentanglementtime,
      title={FreDN: Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition}, 
      author={Zhongde An and Jinhong You and Jiyanglin Li and Yiming Tang and Wen Li and Heming Du and Shouguo Du},
      year={2025},
      eprint={2511.11817},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2511.11817}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This codebase is built upon the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) framework.

## Contact

For questions or issues, please open an issue or contact: [2023213372@stu.sufe.edu.cn]
