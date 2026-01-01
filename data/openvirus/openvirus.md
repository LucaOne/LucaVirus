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
- LucaVirus-Mask
- AI4Bio
- AI4Science
- Nucleotide-Protein
task_categories:
- sequence-modeling
- feature-extraction
size_categories:
- 10M<n<100M
---

# Dataset Card for LucaVirus-OpenVirus-Gene-Prot

## 1. Dataset Summary

**LucaVirus-OpenVirus-Gene-Prot** is the complete, multi-modal **OpenVirus** corpus, curated for the pre-training of the **LucaVirus** biological foundation model. This dataset provides a massive-scale collection of viral sequences, bridging the gap between genomic (nucleotide) and proteomic (protein) data.

The corpus comprises **15.7 million(10.4M nucleotide sequences and 5.2M protein sequences)** non-redundant viral sequences, providing a robust foundation for learning the complex language of viral evolution and the "central dogma" of viral biology.

## 2. Dataset Statistics

| Data Type | Count | `obj_type` Identifier |
| :--- | :--- | :--- |
| **Nucleotide (Genomes)** | 10.4 Million | `gene` |
| **Protein (Amino Acids)** | 5.2 Million | `prot` |
| **Total Sequences** | **15.7 Million** | - |

## 3. Data Structure

The dataset is provided as a compressed **`.tar`** archive. Once extracted, the directory structure follows a standard machine-learning split:

```text
LucaVirus-OpenVirus-Gene-Prot/dataset/v1.0/
├── train/          # Training set (primary corpus for pre-training)
├── dev/            # Validation set (for hyperparameter tuning)
└── test/           # Test set (for final evaluation)
```

Each directory contains one or more **CSV files with headers**.

### Data Schema
Each CSV file includes the following columns:

| Column Name | Description | Details                                                                                                            |
| :--- | :--- |:-------------------------------------------------------------------------------------------------------------------|
| **`obj_id`** | Sample ID | Unique identifier for the sample.                                                                                  |
| **`obj_type`** | Sequence Type | Sequence modality: `gene` (nucleotide) or `prot` (protein).                                                        |
| **`obj_seq`** | Sequence Content | The raw biological sequence (AT(U)GCN for gene; Amino Acids for prot).                                             |
| **`obj_label`** | Label | Metadata, taxonomic info, or functional labels associated with the genome and proteins (Annotation, Bio Knowledge) |

## 4. Dataset Intent

This dataset is specifically designed for:
- **Foundation Model Pre-training**: Building models that can process both DNA/RNA and Protein sequences.
- **Cross-modal Learning**: Understanding the translation and structural relationships within viral biology.
- **Viral Research**: A large-scale benchmark for viral sequence classification, functional annotation, and mutation analysis.

## 5. Usage

### Loading with Python
You can use standard Python libraries to process the data:

```python
import pandas as pd
import tarfile
import os

# Example: Extracting and reading a file
with tarfile.open("LucaVirus-OpenVirus-Gene-Prot.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Gene-Prot/")

with tarfile.open("./LucaVirus-OpenVirus-Gene-Prot/dataset.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Gene-Prot/dataset/")

# Read a specific CSV from the train set
df = pd.read_csv("../LucaVirus-OpenVirus-Gene-Prot/dataset/v1.0/train/3072_train_1.csv")
print(df.head())
```

## 6. Pre-training with LucaVirus
This dataset is the primary source for the **LucaVirus** model family.   
- **Full Corpus (Gene + Prot)**: [LucaVirus-OpenVirus-Gene](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Gene)
- **Protein Subset**: [LucaVirus-OpenVirus-Prot](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Prot)
- **Models**: Visit the [LucaVirus Collection](https://huggingface.co/collections/LucaGroup/lucavirus).


## 7. Citation

If you use this dataset in your research, please cite the following:

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
This dataset is released under the **Apache License 2.0**.

## 9. Contact

*For further information, please visit the [LucaGroup GitHub](https://github.com/LucaOne), email to: [YongHe: sanyuan.hy@alibaba-inc.com, heyongcsat@gmail.com], or contact the team via the Hugging Face organization profile.*


