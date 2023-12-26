#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:20:45 2023
@author: patricksweeney
"""

# =============================================================================
# Imports
# =============================================================================
import torch
import pandas as pd
import numpy as np
from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice import run
import warnings

# =============================================================================
# Setup
# =============================================================================
def setup_environment():
    warnings.filterwarnings("ignore")
    print(torch.__version__)
    if torch.cuda.is_available():
        print(f'CUDA device used: {torch.cuda.get_device_name()}')
        return 'cuda'
    else:
        print('Running tutorial on CPU')
        return 'cpu'

DEVICE = setup_environment()

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_data(url):
    df = pd.read_csv(url, index_col=0)
    df = df.reset_index(drop=True)
    return df

def preprocess_data(df):
    df['idx.id2'].value_counts()
    return df

# =============================================================================
# Dataset Preparation
# =============================================================================
def prepare_datasets(df, item_feat_cols, item_names, encoder):
    item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)

    price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
    nest_dataset = ChoiceDataset(item_index=item_index.clone()).to(DEVICE)
    item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs).to(DEVICE)

    return JointDataset(nest=nest_dataset, item=item_dataset)

# =============================================================================
# Model Initialization
# =============================================================================
def initialize_model(dataset, nest_to_item, encoder):
    for k, v in nest_to_item.items():
        v = [encoder[item] for item in v]
        nest_to_item[k] = sorted(v)

    model = NestedLogitModel(nest_to_item=nest_to_item,
                             nest_formula='',
                             item_formula='(price_obs|constant)',
                             dataset=dataset,
                             shared_lambda=True)
    return model.to(DEVICE)

# =============================================================================
# Model Training
# =============================================================================
def train_model(model, dataset, num_epochs=1000):
    run(model, dataset, num_epochs=num_epochs, model_optimizer="LBFGS")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    data_url = 'https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv'
    df = load_data(data_url)
    df = preprocess_data(df)

    item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
    item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
    encoder = dict(zip(item_names, range(len(item_names))))
    dataset = prepare_datasets(df, item_feat_cols, item_names, encoder)

    nest_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'], 1: ['gc', 'ec', 'er']}
    model = initialize_model(dataset, nest_to_item, encoder)

    print(model)
    train_model(model, dataset)

if __name__ == "__main__":
    main()
