# Official Repository for SDG Benchmark (Under Review)

This repository contains implementations and experiments of Single Domain Generalization (SDG) methods benchmarked on PACS and ImageNet-1k datasets

---

## ⚙️ Environment Setup

Create the required Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate <environment_name>
```

Replace `<environment_name>` with the name specified in your `environment.yml`.

---

## 📁 Dataset Preparation

### 🖼️ PACS Dataset

- Download PACS dataset from the (https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd).
- Rename the domain folders as follows:

| 🗂️ Original Folder Name | 🔁 New Folder Name |
|-------------------------|--------------------|
| art_painting            | AR                 |
| cartoon                 | C                  |
| photo                   | AAP                |
| sketch                  | S                  |

- Specify the dataset's parent directory path in each bash script (`data_dir=`).

### 🧠 ImageNet Dataset

This repository is not optimized for training on large-scale datasets like ImageNet-1k. For large-scale training, we use a separate repository optimized with Distributed Data Parallel (DDP) and multiple GPUs. More details will be added soon. [TODO]

---

## 🚀 Running Experiments

All SDG experiments can be run using bash scripts located in the `bash_scripts` directory.

### ▶️ Example:

Run RandConv method with ResNet18:

```bash
bash bash_scripts/RandConv/run_sweep_Resnet18.sh
```
Results will be saved in Results/PACS_Custom/ME_ADA_CNN/Resnet18

To obtain the average performance from multiple trials, use the provided `Av.py` script 

```bash
python Av.py --filename ./Results/PACS_Custom/ME_ADA_CNN/Resnet18
```
    


---

## 📚 Citations

This repository builds upon these existing repositories:

- [TFS-ViT Token-level Feature Stylization](https://github.com/Mehrdad-Noori/TFS-ViT_Token-level_Feature_Stylization/tree/main)
- [DomainBed by Facebook Research](https://github.com/facebookresearch/DomainBed)

🙏 We thank the original authors for their valuable contributions and code availability.
