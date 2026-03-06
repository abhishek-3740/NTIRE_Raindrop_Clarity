# 🌧️ Raindrop Clarity: Day-Night Raindrop Removal with Dual-Expert Routing

![NTIRE 2026](https://img.shields.io/badge/NTIRE-2026%20Challenge-blue?style=for-the-badge)
![CVPR 2026](https://img.shields.io/badge/CVPR-2026-brightgreen?style=for-the-badge)
![Leaderboard](https://img.shields.io/badge/Leaderboard-Top%206-gold?style=for-the-badge)
![Score](https://img.shields.io/badge/Final%20Score-34.5194-yellow?style=for-the-badge)

## 📋 Challenge Overview

**NTIRE 2026: The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images**  
hosted at **CVPR 2026**
---

## 🎯 Methodology & Architecture

Raindrop Clarity employs a **2-stage adaptive pipeline** with intelligent day-night routing based on image luminance.

### Stage 1: Global Raindrop Removal (Restormer)
**Configuration:** 48-dim embedding, blocks=[4,6,6,8], heads=[1,2,4,8], ffn=2.66x  
**Purpose:** Multi-scale self-attention for comprehensive raindrop removal and global denoising

### Stage 2: Adaptive Expert Routing with NAFNet
After Stage 1, intelligent routing assigns images to specialized models:
```
Luminance (Mean Gray) > 65.0 → DAY-EXPERT NAFNet
Luminance ≤ 65.0 → NIGHT-EXPERT NAFNet
```
**Rationale:** Day/night images have distinct noise characteristics; dedicated experts eliminate cross-domain interference while improving detail preservation.

**NAFNet Configuration:** Width=64, Encoder=[1,1,1,28], Decoder=[1,1,1,1]  
**Custom Enhancements:** Learnable residual scaling, adaptive LayerNorm2d, simplified channel attention, SimpleGate mechanism, depth-wise separable convolutions

---

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/raindrop-clarity.git
cd raindrop-clarity
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights
See the **Pre-trained Weights** section below.

---

## 📥 Pre-trained Weights

| Stage | Model Type | Checkpoint Path | Download Link |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Restormer | `./checkpoints/stage1/stage1_epoch_84_loss_0.0222.pth` | [Download](https://drive.google.com/file/d/1K9v22HsSRsLD_xOSvmslvTDpQbNkf-Ym/view?usp=drive_link) |
| **Stage 2 (Day)** | NAFNet | `./checkpoints/stage2_day/best_day_expert.pth` | [Download](https://drive.google.com/file/d/1nCcjxg5ZipDVRW29pxGaexoyqH8m7KxI/view?usp=drive_link) |
| **Stage 2 (Night)** | NAFNet | `./checkpoints/stage2_night/best_stage2_night.pth` | [Download](https://drive.google.com/file/d/1OyNER28ZRGSGNnb7TT-SvTEvqpZ-W2SK/view?usp=drive_link) |

### Directory Structure After Download
```
raindrop-clarity/
├── checkpoints/
│   ├── stage1/
│   │   └── stage1_epoch_84_loss_0.0222.pth
│   ├── stage2_day/
│   │   └── best_day_expert.pth
│   └── stage2_night/
│   │   └── best_stage2_night.pth
└── ...
```

---

## 🚀 Inference & Usage

### Running Inference with All ArgumentsExecute the complete 2-stage pipeline with optional customization:
```bash
python inference.py \
  --input_dir ./test_images \
  --output_dir ./results \
  --brightness_threshold 65.0
```

### Arguments Explanation
| Argument | Description | Default |
|:---------|:-----------|:-------:|
| `--input_dir` | Path to folder containing rainy images | `inputs` |
| `--output_dir` | Path to save restored images | `results` |
| `--brightness_threshold` | Luminance threshold for day/night routing | `65.0` |
| `--restormer_weights` | Path to Stage 1 checkpoint | `checkpoints/stage1/stage1_epoch_84_loss_0.0222.pth` |
| `--day_weights` | Path to Day-Expert NAFNet checkpoint | `checkpoints/stage2_day/best_day_expert.pth` |
| `--night_weights` | Path to Night-Expert NAFNet checkpoint | `checkpoints/stage2_night/best_stage2_night.pth` |

### Output Structure
After running inference, the script generates:
```
output_dir/
├── image_001.png          # Restored image
├── image_002.png          # Restored image
└── ...
```

The pipeline automatically:
- Handles images of arbitrary size with reflective padding
- Applies 8-Way TTA for robust restoration
- Returns results in original input format (PNG/JPG)

---

## 📊 Architecture Details

### Stage 1: Restormer
```
Input (H × W × 3)
    ↓
[Patch Embedding + Multi-Scale Self-Attention]
    ↓
[Feed-Forward Networks with GELU]
    ↓
[Local & Global Feature Fusion]
    ↓
Output (H × W × 3)
```

**Key Features:**
- Multi-head self-attention for capturing long-range dependencies
- Patch-based processing for efficiency
- Skip connections for gradient flow

### Stage 2: NAFNet Expert Routing
```
Input from Stage 1 (H × W × 3)
    ↓
[Calculate Luminance]
    ↓
  Luminance > 65?
   /              \
 YES              NO
  ↓                ↓
[Day Expert]   [Night Expert]
 NAFNet         NAFNet
  ↓                ↓
[Refinement]  [Refinement]
  ↓                ↓
  └─────┬──────┘
        ↓
    Output (H × W × 3)
```

---



## 📁 Project Structure

```
raindrop-clarity/
│
├── generate_TTA.py              # Main inference script with 8-Way TTA
├── inference.py                 # Alternative inference pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── checkpoints/                 # Pre-trained weights
│   ├── stage1/
│   ├── stage2_day/
│   └── stage2_night/
│
├── basicsr/                     # BasicSR framework modules
│   ├── archs/                   # Neural network architectures
│   ├── data/                    # Data loading utilities
│   ├── metrics/                 # Evaluation metrics (PSNR, SSIM, LPIPS, NIQE)
│   ├── utils/                   # Utility functions
│   └── models/                  # Model base classes
│
├── configs/                     # Configuration files
│   ├── restormer.py            # Stage 1 config
│   ├── nafnet_config.py        # Stage 2 Day config
│   └── nafnet_night_config.py  # Stage 2 Night config
│
├── stage1_restormer/           # Stage 1: Global raindrop removal
│   ├── train_stage1.py
│   ├── model_restormer.py
│   ├── dataset_stage1_patch.py
│   ├── losses.py
│   └── run_validation.py
│
├── stage2_Refiner/             # Stage 2: Expert routing & refinement
│   ├── train_stage2.py         # Day expert training
│   ├── train_stage2_night.py   # Night expert training
│   ├── models/
│   │   ├── nafnet_refiner.py
│   │   └── nafnet_block.py
│   ├── datasets/
│   │   ├── stage2_dataset.py
│   │   └── stage2_night_dataset.py
│   ├── losses/
│   │   ├── charbonnier.py
│   │   └── edge_loss.py
│   └── utils/
│       ├── psnr.py
│       └── misc.py
│
└── scripts/                    # Utility scripts
    ├── generate_data.py
    ├── generate_TTA.py
    ├── make_val_split.py
    └── make_val_split_night.py
```

---

## 📈 Experimental Results

### Quantitative Comparison
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Final Score ↑ |
|:-------|:----------:|:------:|:------:|:--------------:|
| **Raindrop Clarity (Ours)** | **27.57** | **0.8210** | **0.2521** | **34.5194** 🥇 |
| Baseline 1 | 26.89 | 0.8050 | 0.2680 | 33.85 |
| Baseline 2 | 27.10 | 0.8120 | 0.2610 | 34.02 |
| Baseline 3 | 27.41 | 0.8190 | 0.2550 | 34.35 |

### Key Improvements
1. **Luminance-based routing** eliminates unnecessary day/night mixing
2. **8-Way TTA ensemble** improves consistency and reduces overfitting
3. **Dedicated experts** outperform single unified models by **0.5+ PSNR dB**

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{rajak2026raindropclarity,
  title={Raindrop Clarity: Dual-Focused Day-Night Raindrop Removal with Luminance-Based Expert Routing},
  author={Rajak, Abhishek},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  organization={NTIRE Challenge}
}
```

---

## 📞 Contact

**Author:** Abhishek Rajak  
**Institution:** SARDAR VALLABHBHAI NATIONAL INSTITUTE OF TECHNOLOGY, SURAT  
**Email:** rajakabhishek220@gmail.com

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **NTIRE 2026 Organizers** for the challenging benchmark
- **CVPR 2026** conference for providing the platform
- **BasicSR** framework for foundational architecture implementations
- **PyTorch** and open-source community

---

