import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
from datetime import datetime
import torch
from torch import save
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/models')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
import multidcp
import datareader
import metric
import wandb
import pdb
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from multidcp_ae_utils import *
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

# initialize flask app
app = Flask(__name__)
api = Api(app)

# initialize and load model
model = torch.load('./savedmodel_05252022.pt')
model.eval()

# argument parsing
parser = argparse.ArgumentParser(description='MultiDCP AE')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size', type = int)
parser.add_argument('--ae_input_file')
parser.add_argument('--ae_label_file')
parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
parser.add_argument('--max_epoch', type = int)
parser.add_argument('--predicted_result_for_testset', help = "the file directory to save the predicted test dataframe")
parser.add_argument('--hidden_repr_result_for_testset', help = "the file directory to save the test data hidden representation dataframe")
parser.add_argument('--all_cells')
parser.add_argument('--dropout', type=float)
parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                    help = 'whether the cell embedding layer only have linear layers')

args = parser.parse_args()


print('done')