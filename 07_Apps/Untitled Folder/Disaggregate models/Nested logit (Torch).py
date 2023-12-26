#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:20:45 2023

@author: patricksweeney
"""


# =============================================================================
# Import shit
# =============================================================================

import torch_choice
import torch
import numpy as np
import numpy as np
import pandas as pd
import torch
from torch_choice.data import ChoiceDataset, JointDataset

# ignore warnings for nicer outputs.
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice import run
print(torch.__version__)


if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    DEVICE = 'cuda'
else:
    print('Running tutorial on CPU')
    DEVICE = 'cpu'


# =============================================================================
# Import data
# =============================================================================

df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv', index_col=0)
df = df.reset_index(drop=True)
df.head()


# =============================================================================
# Count of choices
# =============================================================================

df['idx.id2'].value_counts()


# =============================================================================
# Choice information
# =============================================================================

item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
num_items = df['idx.id2'].nunique()


# =============================================================================
# Encode choices
# =============================================================================

encoder = dict(zip(item_names, range(num_items)))
item_index = item_index.map(lambda x: encoder[x])
item_index = torch.LongTensor(item_index)


# =============================================================================
# Nesting 
# =============================================================================
