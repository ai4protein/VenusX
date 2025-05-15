import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    EsmTokenizer,
    BertTokenizer,
    AutoTokenizer,
    T5Tokenizer
)

import re
import torch

from models.gvp.util import load_coords
from esm.data import BatchConverter, Alphabet
from typing import Sequence, Tuple, List

class CoordBatchConverter(BatchConverter):

    def __call__(self, 
                 raw_batch: Sequence[Tuple[Sequence, str]], 
                 coords_max_shape, 
                 confidence_max_shape,
                 device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))
        coords_and_confidence, strs, tokens = super().__call__(batch)
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan, max_shape=coords_max_shape)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1., max_shape=confidence_max_shape)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, coord_mask, padding_mask, confidence

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v, max_shape=None):
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(t.shape, max_shape))
            result_i[slices] = t[slices]

        return result


class TokenClsCollateFnForPLM:
    
    def __init__(self, args):

        self.args = args
        self.device = args.device

        if args.plm_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type in ['esm', 'ankh']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type == 'saprot':
            self.tokenizer = EsmTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
        else:
            raise ValueError(f"Unsupported PLM type: {args.plm_type}")

    def __call__(self, batch):

        sequences = [protein["sequence"] for protein in batch]
        labels = [[-100] + protein["label"] + [-100] for protein in batch]
        
        if self.args.plm_type == 'saprot':
            max_len = max([len(seq) // 2 for seq in sequences])
        else:
            max_len = max([len(seq) for seq in sequences])

        if max_len > self.args.max_len: 
            max_len = self.args.max_len

        if self.args.plm_type == 'bert':
            sequences = [" ".join(seq) for seq in sequences]

        if self.args.plm_type == 't5':
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

        if self.args.plm_type == 't5':
            results = self.tokenizer(sequences, add_special_tokens=True, padding="longest")
        else:
            results = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        
        labels_padded = [label[:max_len] + [-100] * (max_len - len(label)) if len(label) < max_len else label[:max_len] for label in labels]
        results["target"] = torch.tensor(labels_padded, dtype=torch.long).to(self.device)
        
        return results

class FragmentClsCollateFnForPLM:

    def __init__(self, args):

        self.args = args
        self.device = args.device

        if args.plm_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type in ['esm', 'ankh']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type == 'saprot':
            self.tokenizer = EsmTokenizer.from_pretrained(args.model_name_or_path)
        elif args.plm_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
        else:
            raise ValueError(f"Unsupported PLM type: {args.plm_type}")

    def __call__(self, batch):

        sequences = [protein["fragment"] for protein in batch]
        labels = [protein["interpro_label"] for protein in batch]
        max_len = max([len(seq) for seq in sequences])

        if max_len > self.args.max_len: 
            max_len = self.args.max_len

        if self.args.plm_type == 'bert':
            sequences = [" ".join(seq) for seq in sequences]

        if self.args.plm_type == 't5':
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

        if self.args.plm_type == 't5':
            results = self.tokenizer(sequences, add_special_tokens=True, padding="longest")
        
        else:

            results = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        
        results["target"] = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return results

class TokenClsCollateFnForProtSSN:
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.pdb_path = args.pdb_file + args.dataset_type + '/raw/'
        self.max_len = args.max_len

    def __call__(self, batch):
        results = {}
        pdbs = [self.pdb_path + protein["interpro"] + '/alphafold2_pdb/' + protein["name"] + '.pdb' for protein in batch]
        results["pdb_file"] = pdbs
        labels = [protein["label"] for protein in batch]
        results["target"] = labels
        
        return results

class FragmentClsCollateFnForProtSSN:

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.pdb_path = args.pdb_file + args.dataset_type + '/raw/'
        self.max_len = args.max_len

    def __call__(self, batch):
        results = {}
        fragment_infor = [
            protein["interpro"] + '/alphafold2_pdb_fragment/' + protein["interpro"] + '_' + protein["name"] + '_' + protein['start'] + '_' + protein['end'] + '.pdb' for protein in batch
        ]
        pdbs = [self.pdb_path + fragment for fragment in fragment_infor]
        results["pdb_file"] = pdbs
        labels = [protein["interpro_label"] for protein in batch]
        results["target"] = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return results

class TokenClsCollateFnForGVP:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.pdb_path = args.pdb_file + args.dataset_type + '/raw/'
        self.max_len = args.max_len

    def __call__(self, batch):
        results = {}
        pdbs = [self.pdb_path + protein["interpro"] + '/alphafold2_pdb/' + protein["name"] + '.pdb' for protein in batch]
        labels = [[-100] + protein["label"] + [-100] for protein in batch]
        
        results['coords'], results['coord_mask'], results['padding_mask'], results['confidence'], max_len = self.load_structure(pdbs)

        labels_padded = [label[:max_len] + [-100] * (max_len - len(label)) if len(label) < max_len else label[:max_len] for label in labels]
        results["target"] = torch.tensor(labels_padded, dtype=torch.long).to(self.device)

        return results
    
    def load_structure(self, pdbs):

        raw_batch = []
        for pdb in pdbs:
            coords, seqs = load_coords(pdb, ['A'])
            raw_batch.append((coords, None, seqs))
        max_len = max([len(coords) for coords, _, _ in raw_batch])
        if max_len > self.max_len:
            max_len = self.max_len
        alphabet = Alphabet.from_architecture("invariant_gvp")
        converter = CoordBatchConverter(alphabet)
        coords, coord_mask, padding_mask, confidence = converter(
            raw_batch=raw_batch,
            coords_max_shape=[max_len, 3, 3],
            confidence_max_shape=[max_len])
        return coords, coord_mask, padding_mask, confidence, max_len

class FragmentClsCollateFnForGVP:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.pdb_path = args.pdb_file + args.dataset_type + '/raw/'
        self.max_len = args.max_len

    def __call__(self, batch):
        results = {}
        fragment_infor = [
            protein["interpro"] + '/alphafold2_pdb_fragment/' + protein["interpro"] + '_' + protein["name"] + '_' + protein['start'] + '_' + protein['end'] + '.pdb' for protein in batch
        ]
        pdbs = [self.pdb_path + fragment for fragment in fragment_infor]
        results['coords'], results['coord_mask'], results['padding_mask'], results['confidence'] = self.load_structure(pdbs)
        results['attention_mask'] = (~results['padding_mask']).long()
        labels = [protein["interpro_label"] for protein in batch]
        results["target"] = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return results
    
    def load_structure(self, pdbs):

        raw_batch = []
        for pdb in pdbs:
            coords, seqs = load_coords(pdb, ['A'])
            raw_batch.append((coords, None, seqs))
        max_len = max([len(coords) for coords, _, _ in raw_batch])
        if max_len > self.max_len:
            max_len = self.max_len
        alphabet = Alphabet.from_architecture("invariant_gvp")
        converter = CoordBatchConverter(alphabet)
        coords, coord_mask, padding_mask, confidence = converter(
            raw_batch=raw_batch,
            coords_max_shape=[max_len, 3, 3],
            confidence_max_shape=[max_len])
        return coords, coord_mask, padding_mask, confidence
