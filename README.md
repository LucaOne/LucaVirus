# LucaVirus
---

# LucaVirus: a Unified Nucleotide-Protein Language Model for Virus

[![PyPI version](https://img.shields.io/badge/pip-v1.1.0-blue)](https://pypi.org/project/lucavirus/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/your-username/LucaVirus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LucaVirus** is a Unified Nucleotide-Protein Language Model for predicting the Evolutionary and Functional Landscapes of Viruses. This repository provides the refactored implementation that is fully compatible with the Hugging Face `transformers` ecosystem, supporting seamless integration for various viral downstream bioinformatics tasks.

## Key Features

- **Hugging Face Native**: Full support for `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoConfig`, and `AutoTokenizer`.
- **Unified Architecture**: Single model architecture handling multiple biological modalities.
- **Task-Specific Heads**:
    - `LucaVirusModel`: For sequences embedding.
    - `LucaVirusForMaskedLM`: For pre-training and sequence recovery.
    - `LucaVirusForSequenceClassification`: For sequence-level tasks (e.g., protein family, solubility, RdRP identity, or promoter prediction).
    - `LucaVirusForTokenClassification`: For residue-level tasks (e.g., secondary structure, binding sites, or post-translational modifications).
- **Extensible**: Easily adaptable to custom downstream tasks using the standard `transformers` API.

## Installation   

```bash
pip install lucavirus==1.1.0
pip install tokenizers==0.19.1
pip install transformers==4.41.2
```

You can install LucaVirus directly from source:
```bash
git clone -b huggingface https://github.com/LucaOne/LucaVirus.git
cd LucaVirus
pip install .
```

For development mode:
```bash
pip install -e .
```

## 🚀Quick Start    

### 1. Feature Extraction/Embedding       
Extract high-dimensional embeddings for downstream analysis or training downstream tasks using LucaVirus-Embedding.    

Please refer to the code in `test/test_lucavirus_embedding.py`.

### 2. MLM Pre-training and Sequence Recovery     
Continue to perform MLM pre-training or sequence recovery.      

Please refer to the code in `test/test_lucavirus_mlm.py`.

### 3. Sequence Classification       
Predict properties for the entire sequence (e.g., RdRP vs. Non-RdRP).   

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.    

Please refer to the code in `test/test_lucavirus_seq_classification.py`.

### 4. Token Classification      
Predict properties for each residue/nucleotide (e.g., Secondary Structure, Binding Sites, Post-Translational Modifications).    

Supports `multi-class classification`, `binary classification`, `multi-label classification`, and `regression` tasks.

Please refer to the code in `test/test_lucavirus_token_classification.py`.

## Model Configuration

| Parameter | Description                         | Default Value              |
| :--- |:------------------------------------|:---------------------------|
| `vocab_size` | Size of the dictionary              | 39                         |
| `hidden_size` | Dimension of the hidden layers      | 2560                       |
| `num_hidden_layers` | Number of Transformer layers        | 12                         |
| `num_attention_heads` | Number of attention heads           | 20                         |
| `position_embeddings` | -                                   | ROPE                       |
| `alphabet` | Type of sequences (e.g., `gene`, `prot`, or `gene_prot`) | `DNA`, `RNA`, and `Protein` |

## Weights Conversion

If you have legacy weights in `.pth` format, use the provided conversion script to migrate them to the Hugging Face format. This script maps the original state dictionary to the new `lucavirus.` prefixed structure.

```bash
python scripts/convert_weights.py     
```

## Citation

If you use LucaVirus in your research, please cite:

```bibtex
@article{lucavirus2025,
  title={Predicting the Evolutionary and Functional Landscapes of Viruses with a Unified Nucleotide-Protein Language Model: LucaVirus.},
  author={Pan, Yuan-Fei* and He, Yong*. et al.},
  journal={bioRxiv},
  year={2025},
  url={https://www.biorxiv.org/content/early/2025/06/20/2025.06.14.659722}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*For more information or issues, please open a GitHub issue or contact the maintainers at [sanyuan.hy@alibaba-inc.com/heyongcsat@gmail.com].*