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
- LucaVirus-Gene
- AI4Bio
- AI4Science
- Nucleotide-Protein
task_categories:
- sequence-modeling
- feature-extraction
size_categories:
- 10M<n<100M
---

# Dataset Card for LucaVirus-OpenVirus-Gene

## 1. Dataset Summary

**LucaVirus-OpenVirus-Gene** is a large-scale genomic dataset consisting exclusively of viral nucleotide sequences. It is a specialized subset of the **OpenVirus** corpus, curated specifically for the pre-training of the **LucaVirus-Gene** foundation model.

By focusing purely on viral genomes, this dataset provides a high-density corpus of **10.4 million** sequences, enabling models to capture the intricate evolutionary patterns, regulatory motifs, and genomic architectures of DNA and RNA viruses.

## 2. Dataset Statistics

The dataset focuses solely on nucleotide sequences (genomes, genes, and fragments):

| Feature | Count / Description |
| :--- | :--- |
| **Total Sequences** | 10.4 Million |
| **Sequence Type** | Nucleotide (DNA/RNA) |
| **`obj_type` Identifier** | `gene` (Exclusive) |
| **Primary Use** | Pre-training for LucaVirus-Gene |

## 3. Data Structure & Format

### 3.1 File Organization
The dataset is provided as a compressed **`.tar`** archive. Upon extraction, the data is partitioned into three standard machine-learning subsets:

```text
LucaVirus-OpenVirus-Gene/dataset/v1.0/
├── train/          # Training set (primary corpus for genomic pre-training)
├── dev/            # Validation set (for model selection and tuning)
└── test/           # Test set (for final evaluation and benchmarking)
```

Each directory (`train`, `dev`, `test`) contains one or more **CSV files** with headers.

### 3.2 CSV Schema
All CSV files follow a consistent four-column schema:

| Column Name | Description | Details                                                                                                |
| :--- | :--- |:-------------------------------------------------------------------------------------------------------|
| **`obj_id`** | Sample ID | Unique identifier for each viral sequence.                                                             |
| **`obj_type`** | Sequence Type | Set to `gene` for all entries in this dataset (Nucleotide).                                            |
| **`obj_seq`** | Sequence Content | Raw nucleotide string (A, T(U), C, G, N).                                                              |
| **`obj_label`** | Label | Metadata, taxonomic info, or functional labels associated with the genome (Annotation, Bio Knowledge). |


## 4. Intended Use

- **Genomic Foundation Modeling**: Building models like **LucaVirus-Gene** that specialize in the "language of genomes."
- **Viral Evolution Studies**: Analyzing conserved nucleotide patterns across divergent viral lineages.
- **Regulatory Element Discovery**: Identifying viral gene boundaries, promoters, and other non-coding functional motifs.

## 5. Usage Example

You can extract the archive and load the genomic data using the following Python snippet:

```python
import tarfile
import pandas as pd
import os

# 1. Extract the genomic dataset
with tarfile.open("LucaVirus-OpenVirus-Gene.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Gene")

with tarfile.open("LucaVirus-OpenVirus-Gene/dataset.tar.gz", "r:gz") as tar:
    tar.extractall(path="./LucaVirus-OpenVirus-Gene/dataset")
    
# 2. Load a sample from the training set
train_path = "./LucaVirus-OpenVirus-Gene/dataset/v1.0/train"
csv_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]

if csv_files:
    # Load the first CSV file
    df = pd.read_csv(os.path.join(train_path, csv_files[0]))
    
    # Verify the sequence type
    print(f"Loaded {len(df)} genomic sequences.")
    print(df[['obj_id', 'obj_seq']].head())
```

## 6. Related Resources

This dataset is a core component of the **LucaGroup** biological modeling ecosystem.
- **Full Corpus (Gene + Prot)**: [LucaVirus-OpenVirus-Gene-Prot](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Gene-Prot)
- **Protein Subset**: [LucaVirus-OpenVirus-Prot](https://huggingface.co/datasets/LucaGroup/LucaVirus-OpenVirus-Prot)
- **Models**: Visit the [LucaVirus Collection](https://huggingface.co/collections/LucaGroup/lucavirus).
- 
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

