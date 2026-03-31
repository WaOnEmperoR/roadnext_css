# RoadNeXt-CSS  
### Enhancing Performance of Context-Based and Full-Stage Features Road Extraction with ConvNeXt on Remote Sensing Imagery

---

## 📌 Overview

Remote sensing technology has advanced to the point where it is easy to acquire images for road extraction. Regional planning, traffic management, disaster management, navigation, and unmanned vehicle trip planning are all areas in which road extraction can be used.

This project proposes **RoadNeXt-CSS**, a deep learning-based method for road extraction from high-resolution remote sensing imagery.

The proposed method is built upon *Road Extraction From Satellite Imagery by Road Context and Full-Stage Feature (RCFSNet)* (DOI: 10.1109/LGRS.2022.3228967). While RCFSNet improves upon DLinkNet using context modeling and full-stage feature fusion, it still relies on conventional convolutional backbones.

**RoadNeXt-CSS enhances RCFSNet by integrating modern backbone architecture and improved training strategies while preserving its core design.**

---

## 📖 Method Background

RCFSNet introduces several key components:

1. **Multi-Scale Context Extraction (MSCE)**  
   Captures rich contextual information at the bottleneck to improve inference capability.

2. **Full-Stage Feature Fusion (FSFF)**  
   Aggregates multi-level features through skip connections to preserve structural details.

3. **Coordinate Dual Attention Mechanism (CDAM)**  
   Enhances feature representation by modeling spatial and channel dependencies.

> ⚠️ **Important**  
> RoadNeXt-CSS **retains all core components** of RCFSNet (MSCE, FSFF, CDAM).  
> The proposed method focuses on **enhancement, not replacement**.

---

## ✨ Proposed Improvements

1. **ConvNeXt Backbone Replacement**
   - Replace ResNet34 with:
     - ConvNeXt-Tiny
     - ConvNeXt-Small
   - Improves feature representation and generalization

2. **Inverse Frequency Weighting (IFW)**
   - Addresses class imbalance (background ≫ road pixels)
   - Improves detection of thin and fragmented roads

3. **Scale-Sensitive Module**
   - Enhances multi-scale feature fusion
   - Improves robustness to varying road widths and occlusions

---

## 🧠 Network Definitions (`/networks`)

This folder contains all model variants.  
All models preserve **MSCE, FSFF, and CDAM**, with variations in backbone, IFW usage, and scale-sensitive module.

### 🔹 Model Variants

| Model | Backbone | IFW | Scale-Sensitive | Output |
|------|----------|-----|-----------------|--------|
| Tiny | ConvNeXt-Tiny | ❌ | ❌ | Sigmoid |
| Tiny_NoSigmoid | ConvNeXt-Tiny | ✅ | ❌ | Logits |
| Tiny_ScaleSen | ConvNeXt-Tiny | ❌ | ✅ | Sigmoid |
| Tiny_NoSigmoid_ScaleSen | ConvNeXt-Tiny | ✅ | ✅ | Logits |
| Small | ConvNeXt-Small | ❌ | ❌ | Sigmoid |
| Small_NoSigmoid | ConvNeXt-Small | ✅ | ❌ | Logits |
| Small_ScaleSen | ConvNeXt-Small | ❌ | ✅ | Sigmoid |
| Small_NoSigmoid_ScaleSen | ConvNeXt-Small | ✅ | ✅ | Logits |

### 📌 Important

- IFW **requires NoSigmoid variants**  
- `pos_weight` must be set in:

```python
double_loss()  # inside /network
```

## 📥 Dataset

The experiments in this project use two publicly available road extraction datasets:

---

### 🔹 Massachusetts Roads Dataset
- Source: https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset  
- Contains high-resolution aerial images with corresponding road annotations  

---

### 🔹 DeepGlobe Roads Dataset
- Source: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset  
- Contains satellite imagery for road extraction  

---

### 🔹 Download Methods

The datasets can be obtained using either:

- **Manual download** from Kaggle  
- **Python script** using Kaggle API  

---

### 🔹 Dataset Structure

After downloading and extracting, organize the datasets as follows:

#### Massachusetts

```
Massachusetts/
├── images/
├── masks/
```

#### DeepGlobe

```
DeepGlobe/
├── images/
├── masks/
├── metadata.csv
```

> 📌 Ensure that image–mask pairs are correctly aligned.

---

### 🔹 DeepGlobe Data Splitting

For the DeepGlobe dataset, the official validation and test sets do not provide annotations.  
Therefore, following the procedure described in the RCFSNet paper:

- **5500 images** are used for training  
- **726 images** are used for testing  
- All samples are selected from the original training set  

---

### 🔹 Splitting Procedure

- Based on `metadata.csv`  
- Use the `split` column with value = `'train'`  
- Perform random sampling with:
  - **Seed = 2024**  
  - Using **pandas library**

#### Example:

```python
import pandas as pd

df = pd.read_csv('metadata.csv')
train_df = df[df['split'] == 'train']

train_sample = train_df.sample(n=5500, random_state=2024)
test_sample = train_df.drop(train_sample.index).sample(n=726, random_state=2024)
```

> ⚠️ Ensure reproducibility by keeping the same random seed (`2024`).
 
## 🏋️ Training

Each dataset directory (`/Massachusetts`, `/DeepGlobe`) contains a `Training/` folder with scripts for running experiments under different configurations.

### 🔹 Naming Format

```bash
rcfsnet_cn<tiny/small>[_scalesen]_<mas/dg2>[_double_pos_<xxx>]_train.py
```

### 🔹 Naming Components

- `cn<tiny/small>` → Backbone (ConvNeXt-Tiny / Small)  
- `scalesen` → Enable Scale-Sensitive module  
- `<mas/dg2>` → Dataset  
- `double_pos_<xxx>` → IFW (pos_weight = xxx, label only)

### 🔹 Training Configurations

#### 1. Baseline
```bash
rcfsnet_cn<tiny/small>_<dataset>_train.py
```

#### 2. IFW
```bash
rcfsnet_cn<tiny/small>_<dataset>_double_pos_<xxx>_train.py
```

#### 3. Scale-Sensitive
```bash
rcfsnet_cn<tiny/small>_scalesen_<dataset>_train.py
```

#### 4. Full (Recommended)
```bash
rcfsnet_cn<tiny/small>_scalesen_<dataset>_double_pos_<xxx>_train.py
```

### 🔹 Important Notes

- IFW requires **NoSigmoid models**  
- `<xxx>` is for **reference only**  
- Set actual value in:

```python
double_loss()
```

### 💾 Model Saving

- Model name must be **manually specified** in training scripts  
- Models are saved in:

```bash
weights/
```

- Recommended naming:
  - `cnsmall_scalesen_dg2_pos10.pth`
  - `cntiny_mas_baseline.pth`
	
## 📊 Evaluation

Each dataset directory contains an `Evaluation/` folder with scripts for model evaluation.

### 🔹 Naming Format

```bash
rcfsnet_cn<tiny/small>[_scalesen][_double]_<mas/dg2>_eval.py
```

### 🔹 Naming Components

- `cn<tiny/small>` → Backbone (ConvNeXt-Tiny / Small)  
- `scalesen` → Use Scale-Sensitive module  
- `double` → Model trained with IFW  
- `<mas/dg2>` → Dataset  

### 🔹 Configurations

- Baseline  
- IFW (`double`)  
- Scale-Sensitive  
- Full (`scalesen + double`)

### ⚙️ Required Configuration

You must manually edit variables in the evaluation script:

#### Load Model
```python
solver.load('../Training/weights/your_model.th')
```

#### Output Folder
```python
target = 'submits/your_experiment/'
```

#### Save CSV
```python
df.to_csv('your_result.csv')
```

### ❗ Important

The values above are **examples only**.  
You must adjust them based on your experiment.

### 📌 Notes

- Ensure consistency between:
  - Training script  
  - Model name  
  - Evaluation script  
  - Output folder  
  - CSV file  
	
## ⚙️ Training Setup

The experiments were conducted on an NVIDIA DGX A100 system with 8 GPUs, where up to 2 GPUs were used simultaneously.

### 🔹 Hyperparameters

- **Optimizer**: Adam  
- **Batch size**: 6  
- **Initial learning rate**: 2 × 10⁻⁴  

### 🔹 Learning Rate Schedule

- The learning rate is reduced by a factor of **1/5** if:
  - Validation accuracy does not improve for **3 consecutive epochs**, and  
  - The learning rate is still greater than **5 × 10⁻⁷**

### 🔹 Early Stopping

- Training stops if there is **no improvement for 6 consecutive epochs**

### 🔹 Training Duration

- **Massachusetts Dataset** → 100 epochs  
- **DeepGlobe Dataset** → 50 epochs  

## 📊 Results — Massachusetts Dataset

| Scenario | Components | IoU | Precision | Recall | F1-Score |
|----------|-----------|-----|----------|--------|----------|
| 0 | RCFSNet | 0.6410 | 0.8709 | 0.7082 | 0.7781 |
| 1 | ConvNeXt-Tiny | 0.6521 | 0.8767 | 0.7186 | 0.7868 |
| 2 | ConvNeXt-Tiny + IFW | 0.6724 | 0.8141 | 0.7932 | 0.7934 |
| 3 | ConvNeXt-Tiny + ScaleSen | 0.6542 | 0.8735 | 0.7228 | 0.7880 |
| 4 | ConvNeXt-Tiny + IFW + ScaleSen | 0.6736 | 0.8215 | 0.7885 | 0.8028 |
| 5 | ConvNeXt-Small | 0.6558 | 0.8850 | 0.7163 | 0.7898 |
| 6 | ConvNeXt-Small + IFW | 0.6746 | 0.8083 | 0.8039 | 0.8037 |
| 7 | ConvNeXt-Small + ScaleSen | 0.6595 | 0.8782 | 0.7258 | 0.7924 |
| 8 | ConvNeXt-Small + IFW + ScaleSen | **0.6779** | 0.8105 | 0.8047 | **0.8061** |

## 📊 Results — DeepGlobe Dataset

| Scenario | Components | IoU | Precision | Recall | F1-Score |
|----------|-----------|-----|----------|--------|----------|
| 0 | RCFSNet | 0.6606 | 0.7855 | 0.8097 | 0.7859 |
| 1 | ConvNeXt-Tiny | 0.6863 | 0.7887 | 0.8409 | 0.8055 |
| 2 | ConvNeXt-Tiny + IFW | 0.6921 | 0.7821 | 0.8595 | 0.8102 |
| 3 | ConvNeXt-Tiny + ScaleSen | 0.6934 | 0.8034 | 0.8361 | 0.8105 |
| 4 | ConvNeXt-Tiny + IFW + ScaleSen | 0.6965 | 0.7950 | 0.8510 | 0.8130 |
| 5 | ConvNeXt-Small | 0.6895 | 0.8063 | 0.8260 | 0.8067 |
| 6 | ConvNeXt-Small + IFW | 0.6974 | 0.7899 | 0.8553 | 0.8121 |
| 7 | ConvNeXt-Small + ScaleSen | 0.6921 | 0.8020 | 0.8360 | 0.8087 |
| 8 | ConvNeXt-Small + IFW + ScaleSen | **0.6984** | 0.7871 | 0.8629 | **0.8146** |

### 📌 Implementation Note

All training settings described above are **fully integrated into the provided training scripts**.  
This ensures consistency and reproducibility of the reported results.

Users may modify these configurations if needed for extended experiments.

## 📦 Pretrained Weights

Pretrained model weights are provided for selected configurations of RoadNeXt-CSS.

> ⚠️ Due to storage limitations of free hosting services, we are currently unable to share all 16 model variants.  
> We apologize for this limitation and provide representative high-performing models instead.

---

### 🔹 Massachusetts Dataset

**ConvNeXt-Tiny + IFW (pos_weight = 5.0) + Scale-Sensitive**  
[![Tiny-Full](https://img.shields.io/badge/Tiny-Full_Model-2ecc71?style=for-the-badge)](https://www.mediafire.com/file/4mdn7d8es97lhoq/RCFSNet_Mas_CnTiny_ScaleSen_double_pos_05_100epoch_v6.th/file)

**ConvNeXt-Small + Scale-Sensitive**  
[![Small-ScaleSen](https://img.shields.io/badge/Small-ScaleSen-f39c12?style=for-the-badge)](https://www.mediafire.com/file/9nfiigbjdsdgyxg/RCFSNet_Mas_CnSmall_ScaleSen_v02_100epoch.th/file)

**ConvNeXt-Small + IFW (pos_weight = 6.0) + Scale-Sensitive**  
[![Small-Full](https://img.shields.io/badge/Small-Full_Model-8e44ad?style=for-the-badge)](https://www.mediafire.com/file/hmnnf38sjucne5z/RCFSNet_Mas_CnSmall_ScaleSen_double_pos_06_100epoch_v9.th/file)

---

### 🔹 DeepGlobe Dataset

**ConvNeXt-Tiny + IFW (pos_weight = 2.5) + Scale-Sensitive**  
[![Tiny-Full](https://img.shields.io/badge/Tiny-Full_Model-2ecc71?style=for-the-badge)](https://www.mediafire.com/file/39jg5nbl7xb623i/RCFSNet_DG_CnTiny_ScaleSen_double_pos_025_100epoch.th/file)

**ConvNeXt-Small + Scale-Sensitive**  
[![Small-ScaleSen](https://img.shields.io/badge/Small-ScaleSen-f39c12?style=for-the-badge)](https://www.mediafire.com/file/fa9s1dl9ef9irdx/RCFSNet_DG2_CnSmall_ScaleSen_50epoch.th/file)

**ConvNeXt-Small + IFW (pos_weight = 3.0) + Scale-Sensitive**  
[![Small-Full](https://img.shields.io/badge/Small-Full_Model-8e44ad?style=for-the-badge)](https://www.mediafire.com/file/u2wwgu0s5sck4xe/RCFSNet_DG2_CnSmall_ScaleSen_double_pos_030_50epoch.th/file)

---

### 🎨 Legend

- 🟢 **Green (Tiny-Full)** → ConvNeXt-Tiny + IFW + Scale-Sensitive  
- 🟠 **Orange (Small-ScaleSen)** → ConvNeXt-Small + Scale-Sensitive  
- 🟣 **Purple (Small-Full)** → ConvNeXt-Small + IFW + Scale-Sensitive  

---

### 📌 Notes

- These models correspond to the **best-performing configurations** reported in the Results section  
- Ensure that the correct architecture (Tiny/Small, IFW, Scale-Sensitive) is used when loading each weight  
- Place downloaded weights in the `weights/` directory before evaluation  