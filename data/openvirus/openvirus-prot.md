---
language:
- en
license: mit
tags:
- Biology
- Bioinformatics
- Virus
- Genomics
- Proteomics
- Nucleotide
- Protein
- Foundation Model
- LucaVirus
- LucaVirus-Prot
- AI4Bio
- AI4Science
- Nucleotide-Protein
task_categories:
- sequence-modeling
- feature-extraction
size_categories:
- 1M<n<10M
---

# Dataset Card for LucaVirus-OpenVirus-Prot

## 1. Dataset Summary

**LucaVirus-OpenVirus-Prot** is a large-scale proteomics dataset consisting exclusively of viral protein sequences. It is a specialized subset of the **OpenVirus** corpus, specifically curated for the pre-training of the **LucaVirus-Prot** (or LucaVirus-Protein) foundation model.

This dataset provides a comprehensive representation of the viral proteosphere, comprising **5.2 million** protein sequences. It is designed to enable biological models to learn the "language of proteins," capturing structural motifs, functional domains, and evolutionary signatures across a vast array of viral families.

## 2. Dataset Statistics

The dataset focuses strictly on amino acid sequences:

| Feature | Count / Description |
| :--- | :--- |
| **Total Sequences** | 5.2 Million |
| **Sequence Type** | Protein (Amino Acids) |
| **`obj_type` Identifier** | `prot` (Exclusive) |
| **Primary Use** | Pre-training for LucaVirus-Prot |

## 3. Data Structure & Format

### 3.1 File Organization
The dataset is distributed as a compressed **`.tar`** archive. Upon extraction, the data is partitioned into three standard machine-learning subsets:

```text
LucaVirus-OpenVirus-Prot/dataset/v1.0/
├── train/          # Training set (primary corpus for protein pre-training)
├── dev/            # Validation set (for model selection and tuning)
└── test/           # Test set (for final evaluation and benchmarking)
```

Each directory (`train`, `dev`, `test`) contains one or more **CSV files** with headers.

### 3.2 CSV Schema
All CSV files follow a consistent four-column schema:

| Column Name | Description | Details                                                                                                |
| :--- | :--- |:-------------------------------------------------------------------------------------------------------|
| **`obj_id`** | Sample ID | Unique identifier for each protein sequence.                                                           |
| **`obj_type`** | Sequence Type | Set to `prot` for all entries in this dataset.                                                         |
| **`obj_seq`** | Sequence Content | Raw amino acid string (standard IUPAC codes).                                                          |
| **`obj_label`** | Annotations | Metadata, taxonomic info, or functional labels associated with the protein (Annotation, Bio Knowledge) |

## 4. Intended Use

- **Protein Foundation Modeling**: Building models like **LucaVirus-Prot** that specialize in understanding protein sequences and their biophysical properties.
- **Functional Annotation**: Developing tools to predict viral protein functions, domains, and active sites.
- **Virus-Host Interaction**: Studying how viral proteins interact with host cellular machinery based on sequence patterns.

## 5. Usage Example

You can extract the archive and load the protein data using the following Python snippet:

```python
import tarfile
import pandas as pd
import os

# 1. Extract the protein dataset
with tarfile.open("LucaVirus-OpenVirus-Prot.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Prot")

with tarfile.open("LucaVirus-OpenVirus-Prot/dataset.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Prot/dataset")

# 2. Load a sample from the training set
train_path = "./LucaVirus-OpenVirus-Prot/dataset/v1.0/train"
csv_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]

if csv_files:
    # Load the first CSV file
    df = pd.read_csv(os.path.join(train_path, csv_files[0]))
    
    # Verify the sequence type
    print(f"Loaded {len(df)} protein sequences.")
    print(df[['obj_id', 'obj_seq', 'obj_label']].head())
```

## 6. Related Resources

This dataset is a core component of the **LucaGroup** biological modeling ecosystem.
- **Full Corpus (Gene + Prot)**: [LucaVirus-OpenVirus-Gene-Prot](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Gene-Prot)
- **Genomic Subset**: [LucaVirus-OpenVirus-Gene](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Gene)
- **Models**: Visit the [LucaVirus Collection](https://huggingface.co/collections/LucaGroup/lucavirus).

## 7. Citation

If you use this dataset in your research, please cite:

```bibtex
@article{lucavirus2025,
  title={Predicting the Evolutionary and Functional Landscapes of Viruses with a Unified Nucleotide-Protein Language Model: LucaVirus.},
  author={Pan, Yuan-Fei* and He, Yong*. et al.},
  journal={bioRxiv},
  year={2025},
  url={https://www.biorxiv.org/content/early/2025/06/20/2025.06.14.659722}
}
```

## 8. License

This dataset is released under the **MIT License**.

## 9. Contact

*For further information, please visit the [LucaGroup GitHub](https://github.com/LucaOne), email to: [YongHe: sanyuan.hy@alibaba-inc.com, heyongcsat@gmail.com], or contact the team via the Hugging Face organization profile.*

