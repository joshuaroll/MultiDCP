from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import argparse
from datetime import datetime
import torch
from torch import save
import numpy as np

# import local modules
sys.path.append('/home/jrollins/home/MultiDCP/MultiDCP/models')
sys.path.append('/home/jrollins/home/MultiDCP/MultiDCP/utils')
import multidcp, datareader, metric, pdb, pickle
from scheduler_lr import step_lr
from loss_utils import apply_NodeHomophily
from tqdm import tqdm
from multidcp_ae_utils import *
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = '/home/jrollins/home/MultiDCP/MultiDCP/website/uploads/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}


# Initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html", content_insert="")
    
if __name__ == "__main__":  
    app.run(host='0.0.0.0',debug=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    content_text = "File upload error"
    filename = ''

    if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        content_text = "File Uploaded!"
        print("Uploaded: " + filename)
        return render_template('predict.html', content_insert=' \n '.join('{}'.format(item) for item in predict(filename)))

    return render_template("upload.html", content_insert=content_text)

def predict(filename):
    output = []
    # check cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output.append("Use GPU: %s" % torch.cuda.is_available())

    # Initialize model
    model_parser = argparse.ArgumentParser(description='MultiDCP AE')	
        
    # Train and dev data are just placeholders here
    model_parser.add_argument('--drug_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/all_drugs_l1000.csv")
    model_parser.add_argument('--gene_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/gene_vector.csv")
    model_parser.add_argument('--train_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/pert_transcriptom/signature_train_cell_1.csv")
    model_parser.add_argument('--dev_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/pert_transcriptom/signature_dev_cell_1.csv")
    # File for inference:
    model_parser.add_argument('--test_file', type=str, default=os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Additional arguments
    model_parser.add_argument('--batch_size', type = int, default=64)
    model_parser.add_argument('--ae_input_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    model_parser.add_argument('--ae_label_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    model_parser.add_argument('--cell_ge_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv", help='the file which used to map cell line to gene expression file')
    model_parser.add_argument('--max_epoch', type = int, default=3)
    model_parser.add_argument('--predicted_result_for_testset', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/teacher_student/teach_stu_perturbedGX.csv", help = "the file directory to save the predicted test dataframe")
    model_parser.add_argument('--hidden_repr_result_for_testset', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/teacher_student/teach_stu_perturbedGX_hidden.csv", help = "the file directory to save the test data hidden representation dataframe")
    model_parser.add_argument('--all_cells', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/ccle_tcga_ad_cells.p")
    model_parser.add_argument('--dropout', type=float, default=0.3)
    model_parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                        help = 'whether the cell embedding layer only have linear layers')

    args, unknown = model_parser.parse_known_args() #args = model_parser.parse_args()

    # Filter content based on what cells we have data for
    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    DATA_FILTER = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422','BRD-U01690642','BRD-U08759356','BRD-U25771771', 'BRD-U33728988', 'BRD-U37049823',
                        'BRD-U44618005', 'BRD-U44700465','BRD-U51951544', 'BRD-U66370498','BRD-U68942961', 'BRD-U73238814',
                        'BRD-U82589721','BRD-U86922168','BRD-U97083655'],
                    "pert_type": ["trt_cp"],
                    "cell_id": all_cells,# ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC'],
                    "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

    # Load data with the autoencoder
    ae_data = datareader.AEDataLoader(device, args)
    data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
    ae_data.setup()
    data.setup()
    output.append('#Train: %d' % len(data.train_data))
    output.append('#Dev: %d' % len(data.dev_data))
    print('#Test: %d' % len(data.test_data))
    output.append('#Train AE: %d' % len(ae_data.train_data))
    output.append('#Dev AE: %d' % len(ae_data.dev_data))
    output.append('#Test AE: %d' % len(ae_data.test_data))

    # parameters initialization
    model_param_registry = initialize_model_registry()
    model_param_registry.update({'num_gene': np.shape(data.gene)[0],
                                'pert_idose_input_dim': len(DATA_FILTER['pert_idose']),
                                'dropout': args.dropout, 
                                'linear_encoder_flag': args.linear_encoder_flag})

    # model creation
    print('--------------with linear encoder: {0!r}--------------'.format(args.linear_encoder_flag))
    model = multidcp.MultiDCP_AE(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained = False)
    model.to(device)
    model = model.double() 

    # initialize empty metrics vectors
    metrics_summary = defaultdict(
    pearson_list_ae_dev = [], pearson_list_ae_test = [], pearson_list_perturbed_dev = [], pearson_list_perturbed_test = [],
    spearman_list_ae_dev = [], spearman_list_ae_test = [], spearman_list_perturbed_dev = [], spearman_list_perturbed_test = [],
    rmse_list_ae_dev = [], rmse_list_ae_test = [], rmse_list_perturbed_dev = [], rmse_list_perturbed_test = [],
    precisionk_list_ae_dev = [], precisionk_list_ae_test = [], precisionk_list_perturbed_dev = [], precisionk_list_perturbed_test = [])
    
    # Load the state dictionary from previously trained model
    model.load_state_dict(torch.load('/home/jrollins/home/MultiDCP/MultiDCP/website/best_multidcp_ae_model_1.pt', map_location = device))

    epoch_loss = 0
    lb_np_ls = []
    predict_np_ls = []
    hidden_np_ls = []
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(tqdm(data.test_dataloader())):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            predict, cells_hidden_repr = model(input_cell_gex=cell_feature, input_drug = drug, 
                                            input_gene = data.gene, mask = mask,
                                            input_pert_idose = pert_idose, job_id = 'perturbed')
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np_ls.append(lb.cpu().numpy()) 
            predict_np_ls.append(predict.cpu().numpy()) 
            hidden_np_ls.append(cells_hidden_repr.cpu().numpy()) 

        lb_np = np.concatenate(lb_np_ls, axis = 0)
        predict_np = np.concatenate(predict_np_ls, axis = 0)
        hidden_np = np.concatenate(hidden_np_ls, axis = 0)
        
        output += test_epoch_end_format(epoch_loss = epoch_loss, lb_np = lb_np, 
                        predict_np = predict_np, steps_per_epoch = i+1, 
                        epoch = i, metrics_summary = metrics_summary,
                        job = 'perturbed', USE_WANDB = False)   

    return output

@app.route("/contact")
def contact():
    return render_template('contact.html')
