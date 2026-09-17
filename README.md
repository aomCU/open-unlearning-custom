<div align="center">

# OpenUnlearning — Slurm & Multilingual Extension
### A resource-conscious extension of OpenUnlearning for Slurm-based experiments and multilingual unlearning

</div>

## 📖 Overview

This repository is based on [OpenUnlearning](https://github.com/locuslab/open-unlearning)
 and extends it with resource-conscious Slurm workflows and support for a custom multilingual English/Thai unlearning dataset.

For the original framework, benchmark documentation, and methodology, please refer to the [OpenUnlearning](https://github.com/locuslab/open-unlearning)
 repository.

---

## 🚀 Running on a Slurm Cluster

We provide specialized, lightweight Slurm scripts optimized for cluster environments under the `slurm/` directory. These are broken down into the standard TOFU benchmark and our custom dataset pipeline.

### 1. TOFU Benchmark Pipeline
Navigate to `slurm/TOFU_BENCHMARK/`:
* `01_tofu_finetune.sh`: Baseline finetuning on the TOFU dataset.
* `02_tofu_unlearn.sh`: Targeted unlearning using specific forget splits.
* `03_tofu_eval.sh`: Manual checkpoint evaluation script.

### 2. Custom Dataset Pipeline
Navigate to `slurm/CUSTOM_DATASET/`:
* `01_mydata_finetune.sh`: Full finetuning using our custom dataset (`all_full.jsonl`).
* `02_mydata_unlearn.sh`: Unlearning specific custom data components.
* `03_mydata_eval.sh`: Manual checkpoint evaluation for the custom dataset.

---

## 📂 Dataset & Naming Conventions

This repository extends the standard TOFU benchmark to support a custom multilingual unlearning dataset (English and Thai). Throughout the codebase, configurations and datasets prefixed with `my` refer to this custom dataset:
* `my.yaml`: Base configuration for the custom multilingual dataset.
* `my_en_forget.yaml`: The English subset designated for unlearning (forget set).
* `my_th_forget.yaml`: The Thai subset designated for unlearning (forget set).
* `my_metrics`: Custom evaluation metrics tailored for this dataset's specific distribution.

### Directory Structure
```text
datasets/
└── mydataset/
    ├── eval/
    │   ├── forget.json     ← my_forget.yaml
    │   └── retain.json     ← my_retain.yaml
    ├── finetune/
    │   ├── all_full.jsonl  ← my_all_full.yaml
    │   ├── en_full.jsonl   ← my_en_full.yaml
    │   └── th_full.jsonl   ← my_th_full.yaml
    ├── source/
    │   └── bilingual_qa.json
    └── unlearn/
        ├── en_forget.json  ← my_en_forget.yaml
        ├── en_retain.json  ← my_en_retain.yaml
        ├── th_forget.json  ← my_th_forget.yaml
        └── th_retain.json  ← my_th_retain.yaml
```

## 📊 Bilingual Dataset Details

This dataset is a bilingual **Thai–English question-answering (QA) dataset** designed for research on **knowledge editing and machine unlearning**. It focuses on factual knowledge about notable individuals from both global and Thai contexts.

## Dataset Construction

### Entity Selection

The dataset contains **200 notable individuals**, equally divided into:

- **100 Global Entities** — internationally recognized individuals
- **100 Thai Entities** — notable individuals from Thailand

Each individual was required to have both **Thai and English Wikipedia pages**. To reduce redundancy in relational information, individuals with direct blood relations (parent–child) or legal relations (spouse) to another individual in the dataset were excluded.

### Data Collection

Personal information was collected primarily from **Wikidata**, supplemented by Thai and English Wikipedia, including infoboxes and article content.

The collected attributes include:

- **Wikidata QID**
- **Name**
- **Date of birth**
- **Place of birth**
- **Nationality**
- **Date of death** (where applicable)
- **Father**
- **Mother**
- **Spouse(s)**

Not all attributes are available for every individual due to missing or undisclosed information.

### Question–Answer Generation

**Gemini 3 Flash** was used to generate natural-language question–answer pairs in both Thai and English from the collected information.

### Quality Control and Data Consistency

All generated question–answer pairs were **manually reviewed** for accuracy, linguistic clarity, and consistency with the source data. However, minor errors or inconsistencies may still remain despite manual review.

The dataset intentionally preserves variations in the original representation of certain attributes, particularly **dates, places and people's names** (including inconsistent punctuation, commas, extra spaces, and alternative spellings or phonetic translations across languages), rather than enforcing a standardized format. These variations may introduce additional challenges for evaluation, especially for exact string matching or other format-sensitive evaluation methods.

### Dataset Split

The data from both Global and Thai Entities is divided equally into:

- **Forget Set** — information intended to be forgotten
- **Retain Set** — information intended to be retained

## Dataset Statistics

The dataset contains **1,240 QA records** derived from 200 individuals.

| Entity Origin | Individuals | Records |
|---|---:|---:|
| Global | 100 | 698 |
| Thai | 100 | 542 |
| **Total** | **200** | **1,240** |

### Data Fields

Each record contains:

- **ID** — unique record identifier
- **Wikidata QID** — Wikidata identifier
- **Entity Origin** — `Global` or `Thai`
- **Attribute Type** — type of factual information
- **Question / Answer** — Thai (`TH`) and English (`EN`) QA pairs
- **Split** — `Forget` or `Retain`

### Attribute Distribution

| Attribute Type | Records | Percentage |
|---|---:|---:|
| Spouse | 235 | 18.95% |
| Nationality | 220 | 17.74% |
| Date of Birth | 198 | 15.97% |
| Place of Birth | 196 | 15.81% |
| Mother | 177 | 14.27% |
| Father | 174 | 14.03% |
| Date of Death | 40 | 3.23% |
| **Total** | **1,240** | **100.00%** |

