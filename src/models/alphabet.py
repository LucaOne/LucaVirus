#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 11:00
@project: LucaVirus
@file: alphabet
@desc: alphabet for LucaVirus
"""
import sys
import numpy as np
import itertools
from typing import Sequence, List
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import gene_seq_replace
except ImportError:
    from src.utils import gene_seq_replace

ATCGU = {"A", "T", "C", "G", "U"}

# gene
gene_prepend_toks = ['[PAD]', '[UNK]']

gene_append_toks = ['[CLS]', '[SEP]', '[MASK]']

gene_standard_toks = ['1', '2', '3', '4', '5', '.', '-', '*']

gene_all_toks = gene_prepend_toks + gene_append_toks + gene_standard_toks

# prot
prot_prepend_toks = ['[PAD]', '[UNK]']

prot_append_toks = ['[CLS]', '[SEP]', '[MASK]']

prot_standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']

prot_all_toks = prot_prepend_toks + prot_append_toks + prot_standard_toks

# gene_prot
gene_prot_prepend_toks = ['[PAD]', '[UNK]']

gene_prot_append_toks = ['[CLS]', '[SEP]', '[MASK]']

gene_prot_standard_toks = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']

gene_prot_all_toks = gene_prot_prepend_toks + gene_prot_append_toks + gene_prot_standard_toks


class Alphabet(object):
    def __init__(
            self,
            standard_toks: Sequence[str],
            prepend_toks: Sequence[str] = gene_prot_prepend_toks,
            append_toks: Sequence[str] = gene_prot_append_toks,
            prepend_bos: bool = True,
            append_eos: bool = True
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.append_toks)
        self.all_toks.extend(self.standard_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["[UNK]"]
        self.padding_idx = self.get_idx("[PAD]")
        self.pad_token_id = self.padding_idx
        self.cls_idx = self.get_idx("[CLS]")
        self.mask_idx = self.get_idx("[MASK]")
        self.eos_idx = self.get_idx("[SEP]")
        self.all_special_tokens = prepend_toks + append_toks
        self.all_special_token_idx_list = [self.tok_to_idx[v] for v in self.all_special_tokens]
        self.unique_no_split_tokens = self.all_toks
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_predefined(cls, name: str):
        if name.lower() == "prot":
            prepend_toks = prot_prepend_toks
            append_toks = prot_append_toks
            standard_toks = prot_standard_toks
        elif name.lower() == "gene":
            prepend_toks = gene_prepend_toks
            append_toks = gene_append_toks
            standard_toks = gene_standard_toks
        elif name.lower() in ["gene_prot", "prot_gene"]:
            prepend_toks = gene_prot_prepend_toks
            append_toks = gene_prot_append_toks
            standard_toks = gene_prot_standard_toks
        else:
            raise ValueError("Not support tokenizer name: %s" % name)
        prepend_bos = True
        append_eos = True

        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    @classmethod
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _tokenize(self, seq) -> str:
        return seq.split()

    def tokenize(self, seq, **kwargs) -> List[str]:
        def split_on_token(tok, seq):
            result = []
            split_seq = seq.split(tok)
            for i, sub_seq in enumerate(split_seq):
                if i < len(split_seq) - 1:
                    sub_seq = sub_seq.rstrip()
                if i > 0:
                    sub_seq = sub_seq.lstrip()

                if i == 0 and not sub_seq:
                    result.append(tok)
                elif i == len(split_seq) - 1:
                    if sub_seq:
                        result.append(sub_seq)
                    else:
                        pass
                else:
                    if sub_seq:
                        result.append(sub_seq)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, seq):
            if not seq.strip():
                return []
            tokenized_seq = []
            seq_list = [seq]
            for tok in tok_list:
                tokenized_seq = []
                for sub_seq in seq_list:
                    if sub_seq not in self.unique_no_split_tokens:
                        tokenized_seq.extend(split_on_token(tok, sub_seq))
                    else:
                        tokenized_seq.append(sub_seq)
                seq_list = tokenized_seq

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_seq
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_seq = split_on_tokens(no_split_token, seq)
        return tokenized_seq

    '''
    def encode(self, seq):
        return [self.tok_to_idx[tok] for tok in self.tokenize(seq)]
    '''

    def encode(self, seq_type, seq):
        if seq_type in ["gene", "dna", "rna", "nucleic_acid", "nucleotide"]:
            if len(ATCGU & set(list(seq.upper()))) > 0:
                seq = gene_seq_replace(seq)
        return [self.tok_to_idx[tok] for tok in self.tokenize(seq)]

    '''
    def encode_for_eval_mask(self, seq):
        return [self.tok_to_idx[tok] if tok != '-' else self.tok_to_idx["[MASK]"] for tok in self.tokenize(seq)]
    '''

    def encode_for_eval_mask(self, seq_type, seq):
        if seq_type in ["gene", "dna", "rna", "nucleic_acid", "nucleotide"]:
            if len(ATCGU & set(list(seq.upper()))) > 0:
                seq = gene_seq_replace(seq)
        return [self.tok_to_idx[tok] if tok != '-' else self.tok_to_idx["[MASK]"] for tok in self.tokenize(seq)]

# gene
# https://www.ncbi.nlm.nih.gov/CBBresearch/Przytycka/download/lectures/PCB_Lect03_Scoring_Matr_Motifs.pdf
# kimura model
nucleotide_point_accepted_alignment_names = ["A", "T", "C", "G"]
alpha = 0.0002
beta = 0.0006
nucleotide_point_accepted_alignment_matrix_prob = [
    [1.0 - 2 * alpha - beta, alpha, alpha, beta],
    [alpha, 1.0 - 2 * alpha - beta, beta, alpha],
    [alpha, beta, 1.0 - 2 * alpha - beta, alpha],
    [beta, alpha, alpha, 1.0 - 2 * alpha - beta]
]


gene_replace_dict = {
    "A": "1",
    "T": "2",
    "C": "3",
    "G": "4",
    "N": "5"
}
gene_vocab_size = len(gene_all_toks)
gene_point_accepted_alignment_matrix_prob = np.eye(N=gene_vocab_size, dtype=float)
for row_idx in range(len(nucleotide_point_accepted_alignment_matrix_prob)):
    row_name = gene_replace_dict[nucleotide_point_accepted_alignment_names[row_idx]]
    new_row_idx = gene_all_toks.index(row_name)
    for col_idx in range(len(nucleotide_point_accepted_alignment_matrix_prob[row_idx])):
        col_name = gene_replace_dict[nucleotide_point_accepted_alignment_names[col_idx]]
        new_col_idx = gene_all_toks.index(col_name)
        gene_point_accepted_alignment_matrix_prob[new_row_idx][new_col_idx] =\
            round(nucleotide_point_accepted_alignment_matrix_prob[row_idx][col_idx], 4)

# prot
# https://iasbs.ac.ir/~vasighi/courses/bioinfo98/bioinfo_09.pdf
aa_point_accepted_alignment_names = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
aa_point_accepted_alignment_matrix_cnt = [
    [9867, 2, 9, 10, 3, 8, 17, 21, 2, 6, 4, 2, 6, 2, 22, 35, 32, 0, 2, 18],
    [1, 9913, 1, 0, 1, 10, 0, 0, 10, 3, 1, 19, 4, 1, 4, 6, 1, 8, 0, 1],
    [4, 1, 9822, 36, 0, 4, 6, 6, 21, 3, 1, 13, 0, 1, 2, 20, 9, 1, 4, 1],
    [6, 0, 42, 9859, 0, 6, 53, 6, 4, 1, 0, 3, 0, 0, 1, 5, 3, 0, 0, 1],
    [1, 1, 0, 0, 9973, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 5, 1, 0, 3, 2],
    [3, 9, 4, 5, 0, 9876, 27, 1, 23, 1, 3, 6, 4, 0, 6, 2, 2, 0, 0, 1],
    [10, 0, 7, 56, 0, 35, 9865, 4, 2, 3, 1, 4, 1, 0, 3, 4, 2, 0, 1, 2],
    [21, 1, 12, 11, 1, 3, 7, 9935, 1, 0, 1, 2, 1, 1, 3, 21, 3, 0, 0, 5],
    [1, 8, 18, 3, 1, 20, 1, 0, 9912, 0, 1, 1, 0, 2, 3, 1, 1, 1, 4, 1],
    [2, 2, 3, 1, 2, 1, 2, 0, 0, 9872, 9, 2, 12, 7, 0, 1, 7, 0, 1, 33],
    [3, 1, 3, 0, 0, 6, 1, 1, 4, 22, 9947, 2, 45, 13, 3, 1, 3, 4, 2, 15],
    [2, 37, 25, 6, 0, 12, 7, 2, 2, 4, 1, 9926, 20, 0, 3, 8, 11, 0, 1, 1],
    [1, 1, 0, 0, 0, 2, 0, 0, 0, 5, 8, 4, 9874, 1, 0, 1, 2, 0, 0, 4],
    [1, 1, 1, 0, 0, 0, 0, 1, 2, 8, 6, 0, 4, 9946, 0, 2, 1, 3, 28, 0],
    [13, 5, 2, 1, 1, 8, 3, 2, 5, 1, 2, 2, 1, 1, 9926, 12, 4, 0, 0, 2],
    [28, 11, 34, 7, 11, 4, 6, 16, 2, 2, 1, 7, 4, 3, 17, 9840, 38, 5, 2 ,2],
    [22, 2, 13, 4, 1, 3, 2, 2, 1, 11, 2, 8, 6, 1, 5, 32, 9871, 0, 2, 9],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 9976, 1, 0],
    [1, 0, 3, 0, 3, 0, 1, 0, 4, 1, 1, 0, 0, 21, 0, 1, 1, 2, 9945, 1],
    [13, 2, 1, 1, 3, 2, 2, 3, 3, 57, 11, 1, 17, 1, 3, 2, 10, 0, 2, 9901]
]
aa_point_accepted_alignment_matrix_prob = []
for idx, name in enumerate(aa_point_accepted_alignment_names):
    total = sum(aa_point_accepted_alignment_matrix_cnt[idx])
    aa_point_accepted_alignment_matrix_prob.append([round(v/total, 4) for v in aa_point_accepted_alignment_matrix_cnt[idx]])

prot_vocab_size = len(prot_all_toks)
prot_point_accepted_alignment_matrix_prob = np.eye(N=prot_vocab_size, dtype=float)
for row_idx in range(len(aa_point_accepted_alignment_matrix_prob)):
    row_name = aa_point_accepted_alignment_names[row_idx]
    new_row_idx = prot_all_toks.index(row_name)
    for col_idx in range(len(aa_point_accepted_alignment_matrix_prob[row_idx])):
        col_name = aa_point_accepted_alignment_names[col_idx]
        new_col_idx = prot_all_toks.index(col_name)
        prot_point_accepted_alignment_matrix_prob[new_row_idx][new_col_idx] = \
            aa_point_accepted_alignment_matrix_prob[row_idx][col_idx]

# gene_prot
gene_prot_vocab_size = len(gene_prot_all_toks)
gene_prot_point_accepted_alignment_matrix_prob = np.eye(N=gene_prot_vocab_size, dtype=float)
for row_idx in range(len(aa_point_accepted_alignment_matrix_prob)):
    row_name = aa_point_accepted_alignment_names[row_idx]
    new_row_idx = gene_prot_all_toks.index(row_name)
    for col_idx in range(len(aa_point_accepted_alignment_matrix_prob[row_idx])):
        col_name = aa_point_accepted_alignment_names[col_idx]
        new_col_idx = gene_prot_all_toks.index(col_name)
        gene_prot_point_accepted_alignment_matrix_prob[new_row_idx][new_col_idx] = \
            aa_point_accepted_alignment_matrix_prob[row_idx][col_idx]


for row_idx in range(len(nucleotide_point_accepted_alignment_matrix_prob)):
    row_name = gene_replace_dict[nucleotide_point_accepted_alignment_names[row_idx]]
    new_row_idx = gene_prot_all_toks.index(row_name)
    for col_idx in range(len(nucleotide_point_accepted_alignment_matrix_prob[row_idx])):
        col_name = gene_replace_dict[nucleotide_point_accepted_alignment_names[col_idx]]
        new_col_idx = gene_prot_all_toks.index(col_name)
        gene_prot_point_accepted_alignment_matrix_prob[new_row_idx][new_col_idx] = \
            round(nucleotide_point_accepted_alignment_matrix_prob[row_idx][col_idx], 4)


if __name__ == "__main__":
    alphabet = Alphabet.from_predefined("gene_prot")
    from src.utils import gene_seq_replace
    print(alphabet.encode(
        seq_type="gene",
        seq=gene_seq_replace("gttgtttggtagctaggagcctgactacatggcttcaaggctaaatggccacaggtgcccaggctatttggcttgctggaggcttcattcat")
    ))

