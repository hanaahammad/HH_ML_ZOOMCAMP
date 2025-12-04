# Hair Type Classification with a Convolutional Neural Network (PyTorch)

This project builds and trains a **Convolutional Neural Network (CNN)** from scratch to classify images of hair into two categories:

- **Curly**
- **Straight**

The workflow follows the structure of a machine learning homework project:  
1) baseline model training,  
2) analysis of results,  
3) continued training with data augmentation.

The implementation uses **PyTorch**, **torchvision**, and GPU acceleration when available.

---

## 🚀 Project Overview

The dataset is already split into:

data/train/
data/test/


We first train a simple CNN for 10 epochs using only basic preprocessing.  
Then we apply **data augmentation** and continue training for 10 additional epochs.

The goal is to analyze how augmentation affects performance, stability, and generalization.

---

## 🧠 Reproducibility

The following settings are used to reduce randomness across runs:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

