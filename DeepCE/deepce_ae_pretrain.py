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
import deepce
import datareader
import metric
import wandb
import pdb
import pickle
from scheduler_lr import step_lr
from loss_utils import apply_NodeHomophily

USE_wandb = True
if USE_wandb:
    wandb.init(project="DeepCE_AE")
else:
    os.environ["WANDB_MODE"] = "dryrun"

start_time = datetime.now()

parser = argparse.ArgumentParser(description='DeepCE PreTraining')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--unfreeze_steps', help='The epochs at which each layer is unfrozen, like <<1,2,3,4>>')
parser.add_argument('--ae_input_file')
parser.add_argument('--ae_label_file')

args = parser.parse_args()

drug_file = args.drug_file
gene_file = args.gene_file
dropout = float(args.dropout)
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
unfreeze_steps = args.unfreeze_steps.split(',')
assert len(unfreeze_steps) == 4, "number of unfreeze steps should be 4"
unfreeze_pattern = [False, False, False, False]
ae_input_file = args.ae_input_file
ae_label_file = args.ae_label_file

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 32
cell_decoder_dim = 2145 # autoencoder label's dimension
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg' # 'mse_and_homophily'
intitializer = torch.nn.init.kaiming_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          #"cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "cell_id": ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

ae_data = datareader.AEDataReader(ae_input_file, ae_label_file, device)
data = datareader.DataReader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                             gene_expression_file_test, filter, device)
print('#Train: %d' % len(data.train_feature['drug']))
print('#Dev: %d' % len(data.dev_feature['drug']))
print('#Test: %d' % len(data.test_feature['drug']))

# model creation
model = deepce.DeepCE_AE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=978, 
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, cell_decoder_dim = cell_decoder_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.to(device)
model = model.double()
if USE_wandb:
     wandb.watch(model, log="all")

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: step_lr([int(x) for x in unfreeze_steps], x)])
best_dev_loss = float("inf")
best_dev_pearson = float("-inf")
pearson_list_ae_dev = []
pearson_list_ae_test = []
pearson_list_perturbed_dev = []
pearson_list_perturbed_test = []
spearman_list_ae_dev = []
spearman_list_ae_test = []
spearman_list_perturbed_dev = []
spearman_list_perturbed_test = []
rmse_list_ae_dev = []
rmse_list_ae_test = []
rmse_list_perturbed_dev = []
rmse_list_perturbed_test = []
precisionk_list_ae_dev = []
precisionk_list_ae_test = []
precisionk_list_perturbed_dev = []
precisionk_list_perturbed_test = []
pearson_raw_list_ae = []
pearson_raw_list_perturbed = []
for epoch in range(max_epoch):
    
    scheduler.step()
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))
    if str(epoch) in unfreeze_steps:
        number_layer_to_unfreeze = 3 - unfreeze_steps[::-1].index(str(epoch)) ## find the position of last occurance of number epoch
        for i in range(3-number_layer_to_unfreeze,4):
            unfreeze_pattern[i] = True 
        model.gradual_unfreezing(unfreeze_pattern)
    
    print(unfreeze_pattern)
    print("Iteration %d:" % (epoch+1))
    model.train()
    
    epoch_loss = 0

    for i, (feature, label, cell_type) in enumerate(ae_data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):
        
        optimizer.zero_grad()
        #### the auto encoder step doesn't need other input rather than feature
        predict, cell_hidden_ = model(input_drug=None, input_gene=None, mask=None, input_pert_type=None, 
                        input_cell_id=feature, input_pert_idose=None, job_id = 'ae', epoch = epoch)
        #loss = approxNDCGLoss(predict, lb, padded_value_indicator=None)
        loss = model.loss(label, predict)
        loss_2 = apply_NodeHomophily(cell_hidden_, cell_type, device = device)
        loss_t = loss + 0.5 * loss_2
        loss_t.backward()
        print(loss.item(), loss_2.item())
        optimizer.step()
        epoch_loss += loss.item()
    
    print('AE Train loss:')
    print(epoch_loss/(i+1))
    wandb.log({'AE Train loss': epoch_loss/(i+1)}, step = epoch)

    model.eval()

    epoch_loss = 0
    lb_np = np.empty([0, cell_decoder_dim])
    predict_np = np.empty([0, cell_decoder_dim])
    with torch.no_grad():
        for i, (feature, label, _) in enumerate(ae_data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False)):
            predict = model(input_drug=None, input_gene=None, mask=None, input_pert_type=None, 
                        input_cell_id=feature, input_pert_idose=None, job_id = 'ae', epoch = epoch)
            loss = model.loss(label, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

        print('AE Dev loss:')
        print(epoch_loss / (i + 1))
        wandb.log({'AE Dev loss': epoch_loss/(i+1)}, step=epoch)
        rmse = metric.rmse(lb_np, predict_np)
        rmse_list_ae_dev.append(rmse)
        print('AE RMSE: %.4f' % rmse)
        wandb.log({'AE Dev RMSE': rmse}, step=epoch)
        pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_list_ae_dev.append(pearson)
        print('AE Pearson\'s correlation: %.4f' % pearson)
        wandb.log({'AE Dev Pearson': pearson}, step = epoch)
        spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_list_ae_dev.append(spearman)
        print('AE Spearman\'s correlation: %.4f' % spearman)
        wandb.log({'AE Dev Spearman': spearman}, step = epoch)
        ae_precision = []
        for k in precision_degree:
            precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
            print("AE Precision@%d Positive: %.4f" % (k, precision_pos))
            print("AE Precision@%d Negative: %.4f" % (k, precision_neg))
            # wandb.log({'AE Dev Precision Positive@{0!r}'.format(k): precision_pos}, step = epoch)
            # wandb.log({'AE Dev Precision Negative@{0!r}'.format(k): precision_neg}, step = epoch)
            ae_precision.append([precision_pos, precision_neg])
        precisionk_list_ae_dev.append(ae_precision)

        if best_dev_pearson < pearson:
            best_dev_pearson = pearson
            save(model.sub_deepce.state_dict(), 'best_sub_deepce_storage_with_noise_')

    epoch_loss = 0
    lb_np = np.empty([0, cell_decoder_dim])
    predict_np = np.empty([0, cell_decoder_dim])
    with torch.no_grad():
        for i, (feature, label, _) in enumerate(ae_data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
            predict = model(input_drug=None, input_gene=None, mask=None, input_pert_type=None, 
                        input_cell_id=feature, input_pert_idose=None, job_id = 'ae')
            loss = model.loss(label, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('AE Test loss:')
        print(epoch_loss / (i + 1))
        wandb.log({'AE Test Loss': epoch_loss / (i + 1)}, step = epoch)
        rmse = metric.rmse(lb_np, predict_np)
        rmse_list_ae_test.append(rmse)
        print('AE RMSE: %.4f' % rmse)
        wandb.log({'AE Test RMSE': rmse} , step = epoch)
        pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_list_ae_test.append(pearson)
        print('AE Pearson\'s correlation: %.4f' % pearson)
        wandb.log({'AE Test Pearson': pearson}, step = epoch)
        spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_list_ae_test.append(spearman)
        print('AE Spearman\'s correlation: %.4f' % spearman)
        wandb.log({'AE Test Spearman': spearman}, step = epoch)
        ae_precision_test = []
        for k in precision_degree:
            precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
            print("AE Precision@%d Positive: %.4f" % (k, precision_pos))
            print("AE Precision@%d Negative: %.4f" % (k, precision_neg))
            # wandb.log({'AE Test Precision Positive@{0!r}'.format(k): precision_pos}, step=epoch)
            # wandb.log({'AE Test Precision Negative@{0!r}'.format(k): precision_neg}, step=epoch)
            ae_precision_test.append([precision_pos, precision_neg])
        precisionk_list_ae_test.append(ae_precision_test)

best_dev_epoch = np.argmax(pearson_list_ae_dev)
print("Epoch %d got best AE Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, pearson_list_ae_dev[best_dev_epoch]))
print("Epoch %d got AE Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, spearman_list_ae_dev[best_dev_epoch]))
print("Epoch %d got AE RMSE on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_ae_dev[best_dev_epoch]))
print("Epoch %d got AE P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_ae_dev[best_dev_epoch][-1][0],
                                                                  precisionk_list_ae_dev[best_dev_epoch][-1][1]))

print("Epoch %d got AE Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_list_ae_test[best_dev_epoch]))
print("Epoch %d got AE Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_list_ae_test[best_dev_epoch]))
print("Epoch %d got AE RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_ae_test[best_dev_epoch]))
print("Epoch %d got AE P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_ae_test[best_dev_epoch][-1][0],
                                                                  precisionk_list_ae_test[best_dev_epoch][-1][1]))


best_test_epoch = np.argmax(pearson_list_ae_test)
print("Epoch %d got AE best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, pearson_list_ae_test[best_test_epoch]))
print("Epoch %d got AE Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, spearman_list_ae_test[best_test_epoch]))
print("Epoch %d got AE RMSE on test set: %.4f" % (best_test_epoch + 1, rmse_list_ae_test[best_test_epoch]))
print("Epoch %d got AE P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                  precisionk_list_ae_test[best_test_epoch][-1][0],
                                                                  precisionk_list_ae_test[best_test_epoch][-1][1]))

end_time = datetime.now()
print(end_time - start_time)
