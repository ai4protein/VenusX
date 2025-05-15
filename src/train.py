import gc
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import os
import csv
from accelerate import Accelerator
from copy import deepcopy
import yaml
from dataset import VenusDataModule
from metrics import VenusMetrics
from models.pooling import VenusClassification

from models.plm import PLMEncoder
from models.gvp.encoder import GVPEncoder, gvp_config
from models.protssn.models import PLM_model, GNN_model
from models.protssn.encoder import ProtSSN, protssn_config
from models.protssn.dataset_utils import NormalizeProtein

import torch.multiprocessing as mp

class Seed():
    
    def __init__(self, seed_value):
        super(Seed, self).__init__()
        self.seed_value = seed_value
    
    def set(self):
        
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(self.seed_value)
            torch.cuda.manual_seed_all(self.seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            
        return f'Seed has been set with value {self.seed_value}.'
    
class EarlyStopping:
    
    def __init__(self, 
                 patience=10,
                 larger_better=True):
        
        self.patience = patience
        self.counter = 0
        self.larger_better = larger_better
        if larger_better:
            self.best = -np.inf
        else:
            self.best = np.inf

    def early_stopping(self, current_indicator):
        
        update = (current_indicator > self.best) if self.larger_better else (current_indicator < self.best)
        
        if update:
            self.best = current_indicator
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class Trainer(nn.Module):

    def __init__(self, args, model):
        

        super().__init__()
        
        self.args = args
        Seed(args.seed).set()
        self.model = model
        self.device = args.device
        self.epoch = args.epoch
        self.task_type = {'token_cls': 'binaryclass', 'fragment_cls': 'multiclass'}.get(args.task, KeyError)
        self.metrics = VenusMetrics(
            args.num_labels, args.task, self.task_type, args.label_skew, args.device
        )
        self.es = EarlyStopping(args.patience)
        
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.optimizer =  torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.init_lr)
        if args.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epoch, eta_min=args.min_lr)
        
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e3
        all_params = sum(p.numel() for p in model.parameters()) // 1e6
        if all_params < 1:
            all_params = all_params // 1e3
        print(f"Trainable parameters: {train_params}k; All parameters: {all_params}M")

        self.log = self.init_log(self.task_type)

        self.best_epoch = 0
        self.best_state = None

    def init_log(self, task_type):
        if task_type == 'regression':
            return {
                'train_loss': [], 
                'val_loss': [], 'val_spearman': [], 'val_mse': [], 'val_mae': [], 'val_r2': [],
                'test_spearman': [], 'test_mse': [], 'test_mae': [], 'test_r2': []
            }
        elif task_type == 'multilabel':
            return {
                'train_loss': [], 
                'val_loss': [], 'val_aupr': [], 'val_f1max': [],
                'test_aupr': [], 'test_f1max': []
            }
        elif task_type == 'multiclass':
            return {
                    'train_loss': [], 
                    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_mcc': [],
                    'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_mcc': []
                }
        elif task_type == 'binaryclass':
            if self.args.label_skew:
                return {
                    'train_loss': [], 'val_loss': [], 
                    'val_aupr': [], 'val_recall_1': [], 'val_precision_1': [], 'val_f1_1': [], 
                    'val_classwise_f1_0': [], 'val_classwise_f1_1': [], 'val_macro_f1': [], 
                    'test_aupr': [], 'test_recall_1': [], 'test_precision_1': [], 'test_f1_1': [], 
                    'test_classwise_f1_0': [], 'test_classwise_f1_1': [], 'test_macro_f1': []
                }
            else:
                return {
                    'train_loss': [], 
                    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_mcc': [],
                    'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_mcc': []
                }
                
    
    def forward(self, data_module):
        
        print("Start training...")
        train_dl, val_dl, test_dl = data_module()
        self.model, self.optimizer, train_dl = self.accelerator.prepare(
            self.model, self.optimizer, train_dl
        )

        for epoch in range(self.epoch):
            self.model.train()
            tl = []; vl = [] # loss
            with tqdm(total=len(train_dl)) as pbar:
                pbar.set_description(f'Training Epoch {epoch+1}/{self.epoch}')
                for batch_idx, batch in enumerate(train_dl):
                    tl.append(self.train_step(batch))
                    pbar.set_postfix({'current loss': sum(tl)/len(tl)})
                    pbar.update(1)
            self.update_log('train', tl)
            print(f">> Epoch {epoch+1} Loss: {sum(tl)/len(tl)}")

            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dl):
                    vl.append(self.val_step(batch))
                metrics_res = self.metrics.compute()
                self.metrics.reset()
                self.update_log('val', vl, metrics_res)
                print(f">> Valid loss: {sum(vl)/len(vl)}")
                self.print_metrics_res(metrics_res)
                if self.task_type in ['multiclass', 'binaryclass']:
                    if self.args.label_skew:
                        key_metric = 'aupr'
                    else:
                        key_metric = 'acc'
                elif self.task_type == 'multilabel':
                    key_metric = 'f1max'
                elif self.task_type == 'regression':
                    key_metric = 'spearman'
                if metrics_res[key_metric] >= max(self.log[f'val_{key_metric}']):
                    print(f'>> Save best model at epoch {epoch+1}')
                    self.best_state = deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1
            if self.es.early_stopping(metrics_res[key_metric]): 
                print(f'>> Early stop at epoch {epoch+1}'); break 

        print(f'>> Training finished.\n>> Testing...')
        print(f'>> Best epoch: {self.best_epoch}')
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                self.test_step(batch)
            metrics_res = self.metrics.compute()
            self.update_log('test', metrics=metrics_res)
            self.print_metrics_res(metrics_res)
        
        self.save_log()

        gc.collect()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.model):
            loss = self.model(batch).loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.args.lr_scheduler:
                self.scheduler.step()
            return round(loss.item(), 4)

    def val_step(self, batch):
        output = self.model(batch)
        self.metrics.update(output.logits, batch["target"])
        return round(self.model(batch).loss.item(), 4)
    
    def test_step(self, batch):
        self.metrics.update(
            self.model(batch).logits, 
           batch["target"]
        )
    
    def print_metrics_res(self, res):
        print("== Metrics:")
        if self.task_type == 'regression':
            print(f"== Spearman: {round(res['spearman'].item(), 4)}")
            print(f"== MSE: {round(res['mse'].item(), 4)}")
            print(f"== MAE: {round(res['mae'].item(), 4)}")
            print(f"== R2: {round(res['r2'].item(), 4)}")

        elif self.task_type == 'multilabel':
            print(f"== AUPR: {round(res['aupr'].item(), 4)}")  
            print(f"== F1-max: {round(res['f1max'].item(), 4)}")

        elif self.task_type == 'multiclass':
            print(f"== Accuracy: {round(res['acc'].item(), 4)}")
            print(f"== Precision: {round(res['precision'].item(), 4)}")
            print(f"== Recall: {round(res['recall'].item(), 4)}")
            print(f"== F1-score: {round(res['f1'].item(), 4)}")
            print(f"== MCC: {round(res['mcc'].item(), 4)}")

        elif self.task_type == 'binaryclass':
            if self.args.label_skew:
                print(f"== AUPR: {round(res['aupr'].item(), 4)}")
                print(f"== Precision: {round(res['precision_1'].item(), 4)}")
                print(f"== Recall: {round(res['recall_1'].item(), 4)}")
                print(f"== F1-score: {round(res['f1_1'].item(), 4)}")
                print(f"== Classwise F1 on 0: {round(res['classwise_f1_0'].item(), 4)}")
                print(f"== Classwise F1 on 1: {round(res['classwise_f1_1'].item(), 4)}")
                print(f"== Macro F1: {round(res['macro_f1'].item(), 4)}")
            else:
                print(f"== Accuracy: {round(res['acc'].item(), 4)}")
                print(f"== Precision: {round(res['precision'].item(), 4)}")
                print(f"== Recall: {round(res['recall'].item(), 4)}")
                print(f"== F1-score: {round(res['f1'].item(), 4)}")
                print(f"== MCC: {round(res['mcc'].item(), 4)}")

    def update_log(self, phase, loss=None, metrics=None):

        if phase in ['train', 'val'] and loss is not None:
            self.log[f'{phase}_loss'].extend(loss)

        if metrics is not None:
            if self.task_type == 'regression':
                self.log[f'{phase}_spearman'].append(metrics['spearman'].item())
                self.log[f'{phase}_mse'].append(metrics['mse'].item())
                self.log[f'{phase}_mae'].append(metrics['mae'].item())
                self.log[f'{phase}_r2'].append(metrics['r2'].item())
            elif self.task_type == 'multilabel':
                self.log[f'{phase}_aupr'].append(metrics['aupr'].item())
                self.log[f'{phase}_f1max'].append(metrics['f1max'].item())
            elif self.task_type == 'multiclass':
                self.log[f'{phase}_acc'].append(metrics['acc'].item())
                self.log[f'{phase}_precision'].append(metrics['precision'].item())
                self.log[f'{phase}_recall'].append(metrics['recall'].item())
                self.log[f'{phase}_f1'].append(metrics['f1'].item())
                self.log[f'{phase}_mcc'].append(metrics['mcc'].item())
            elif self.task_type == 'binaryclass':
                if self.args.label_skew:
                    self.log[f'{phase}_aupr'].append(metrics['aupr'].item())
                    self.log[f'{phase}_recall_1'].append(metrics['recall_1'].item())
                    self.log[f'{phase}_precision_1'].append(metrics['precision_1'].item())
                    self.log[f'{phase}_f1_1'].append(metrics['f1_1'].item())
                    self.log[f'{phase}_classwise_f1_0'].append(metrics['classwise_f1_0'].item())
                    self.log[f'{phase}_classwise_f1_1'].append(metrics['classwise_f1_1'].item())
                    self.log[f'{phase}_macro_f1'].append(metrics['macro_f1'].item())
                else:
                    self.log[f'{phase}_acc'].append(metrics['acc'].item())
                    self.log[f'{phase}_precision'].append(metrics['precision'].item())
                    self.log[f'{phase}_recall'].append(metrics['recall'].item())
                    self.log[f'{phase}_f1'].append(metrics['f1'].item())
                    self.log[f'{phase}_mcc'].append(metrics['mcc'].item())
    
    def save_log(self):
        
        self.file_header = self.args.csv_log_path + self.args.task + '/' + self.args.encoder_type + '/'
        
        parts = self.args.dataset_file.strip('/').split('/')
        if parts[-1].startswith('sim_'):
            dataset_type = '_'.join(parts[-3:])
        else:
            dataset_type = '_'.join(parts[-2:])

        if self.args.encoder_type == 'plm':
            file_name = os.path.join(self.file_header, self.args.plm_dir.split('/')[-1] + '_' + dataset_type + '.csv')
        else:
            file_name = os.path.join(self.file_header,  dataset_type + '.csv')


        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            task_headers = {
                'regression': ['val_spearman', 'val_mse', 'val_mae', 'val_r2',
                            'test_spearman', 'test_mse', 'test_mae', 'test_r2'],
                'multilabel': ['val_aupr', 'val_f1max', 'test_aupr', 'test_f1max'],
                'multiclass': ['val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_mcc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_mcc'],
                'binaryclass': ['val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_mcc',
                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_mcc'] if not self.args.label_skew else
                ['val_aupr', 'val_recall_1', 'val_precision_1', 'val_f1_1',
                'val_classwise_f1_0', 'val_classwise_f1_1', 'val_macro_f1',
                'test_aupr', 'test_recall_1', 'test_precision_1', 'test_f1_1',
                'test_classwise_f1_0', 'test_classwise_f1_1', 'test_macro_f1']
            }

            metric_headers = task_headers[self.task_type]

            header = ['step/epoch', 'train_loss', 'val_loss'] + metric_headers
            writer.writerow(header)

            max_length = max(len(self.log['train_loss']), len(self.log['val_loss']), 
                            *[len(self.log[m]) for m in metric_headers])
            for i in range(max_length):
                row = [i + 1, 
                    self.log['train_loss'][i] if i < len(self.log['train_loss']) else '',
                    self.log['val_loss'][i] if i < len(self.log['val_loss']) else '']

                for metric in metric_headers:
                    row.append(self.log[metric][i] if i < len(self.log[metric]) else '')

                writer.writerow(row)
        print(f'>> Log saved to {file_name}')

def create_parser():

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--task', type=str, choices=['token_cls', 'fragment_cls'], default='token_cls')
    parser.add_argument('--dataset_type', type=str, choices=['domain', 'motif', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--label_skew', action="store_true", default=False)
    # train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr_scheduler', action="store_true", default=False)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--early_stopping', action="store_true", default=True)
    parser.add_argument('--patience', type=int, default=10)
    # encoder
    parser.add_argument('--encoder_type', type=str, choices=['plm', 'gvp', 'protssn'], default='plm')
    # plm
    parser.add_argument('--plm_type', type=str, choices=['esm', 'bert', 'ankh', 'saprot', 'prosst'])
    parser.add_argument('--plm_dir', type=str)
    parser.add_argument('--plm_freeze', action="store_true", default=True)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--foldseek', type=str)
    parser.add_argument('--fs_process_id', type=int, default=0)
    # pooling
    parser.add_argument('--hidden_size', type=float, default=128)
    parser.add_argument('--attention_pooling', action="store_true", default=True)
    parser.add_argument('--pooling_method', type=str, 
                        choices=['mean', 'pooling_method', 'light_attention'], default='mean')
    parser.add_argument('--pooling_dropout', type=float, default=0.1)
    # output
    parser.add_argument('--num_labels', type=int, default=1)
    # csv_log
    parser.add_argument('--csv_log_path', type=str)

    return parser.parse_args()

def create_args_from_dict(param_dict):
    args_list = []
    for k, v in param_dict.items():
        args_list.append(f'--{k}')
        args_list.append(str(v))

    parser = argparse.ArgumentParser()
    for k, v in param_dict.items():
        arg_type = type(v)
        parser.add_argument(f'--{k}', type=arg_type)

    args = parser.parse_args(args_list)
    return args

if __name__ == '__main__':

    mp.set_start_method("spawn")

    args = create_parser()
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    
    data_module = VenusDataModule(args, train=True)

    if args.encoder_type == 'plm':
        encoder = PLMEncoder(args)
    elif args.encoder_type == 'gvp':
        encoder = GVPEncoder(create_args_from_dict(gvp_config))
    elif args.encoder_type == 'protssn':
        # ===================================================================================
        protssn_args = create_args_from_dict(protssn_config)
        protssn_args.gnn_config = yaml.load(open(protssn_args.gnn_config), Loader=yaml.FullLoader)[protssn_args.gnn]
        protssn_args.gnn_config["hidden_channels"] = protssn_args.gnn_hidden_dim
        plm_model = PLM_model(protssn_args)
        gnn_model = GNN_model(protssn_args)
        gnn_model.load_state_dict(torch.load(protssn_args.gnn_model_path))
        encoder = ProtSSN(
            c_alpha_max_neighbors=protssn_args.c_alpha_max_neighbors,
            pre_transform=NormalizeProtein(
                filename=f'src/models/protssn/cath_k{protssn_args.c_alpha_max_neighbors}_mean_attr.pt'
            ),
            plm_model=plm_model, gnn_model=gnn_model
        )
        # ====================================================================================
    
    model = VenusClassification(args, encoder=encoder)

    Trainer(args, model.to(args.device))(data_module)