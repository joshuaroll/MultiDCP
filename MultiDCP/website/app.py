from flask import Flask, redirect, url_for, render_template, request, send_file
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
from ranking_list import *
import warnings
import requests
import responses
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = '/home/jrollins/home/MultiDCP/MultiDCP/website/uploads/'
RESULTS_FOLDER = '/home/jrollins/home/MultiDCP/MultiDCP/website/results/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}


# Initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

@app.route("/")
def home():
    return render_template("index.html", content_insert=drug_info("tylenol"))

# Get information from DrugBank
def drug_info(drug_name):
    r = requests.get("https://www.dgidb.org/api/v2/interactions.json?drugs=" + drug_name)
    return r.text
    
if __name__ == "__main__":  
    app.run(host='0.0.0.0',debug=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/table/<path:csvname>', methods=['GET', 'POST'])
def table(csvname):
    # converting csv to html
    results = os.path.join(app.root_path, app.config['RESULTS_FOLDER'])
    data = pd.read_csv(results + csvname)
    return render_template('table.html', tables=[data.to_html()], titles=[''])
  


@app.route('/results/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    results = os.path.join(app.root_path, app.config['RESULTS_FOLDER'])
    print(results)
    return send_file(results + filename, attachment_filename=filename)


# test file 'website/results/results_df_072422.csv'
@app.route('/rank/<path:filename>', methods=['GET', 'POST'])
def rank(filename):
    rank_file = os.path.join(app.root_path, app.config['RESULTS_FOLDER']) + filename
    ph_topsigs, ph_topinfo, PH_DATA_FILE, PH_RANKED_FILE, PH_TOPSIG_FILE = rank_results(rank_file)
    return render_template("ranked.html", topsigs=ph_topsigs, topinfo=ph_topinfo, data_filename=PH_DATA_FILE, ranked_filename=PH_RANKED_FILE, topsig_filename=PH_TOPSIG_FILE) # old returns file/possibly use later send_file(results + filename, attachment_filename=filename)


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file_sig = request.files['sig_file']
    uploaded_file_ge = request.files['basal_ge_file']
    content_text = "File upload error"
    filename_sig, filename_ge = '', ''

    if uploaded_file_sig.filename != '' and allowed_file(uploaded_file_sig.filename): #\
    #and uploaded_file_ge.filename != '' and allowed_file(uploaded_file_ge.filename):
        filename_sig = secure_filename(uploaded_file_sig.filename)
        uploaded_file_sig.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_sig))
        #filename_ge = secure_filename(uploaded_file_ge.filename)
        #uploaded_file_ge.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_ge))

        content_text = "File Uploaded!"
        filename_ge = 'adjusted_ccle_tcga_ad_tpm_log2.csv'
        print("Uploaded: " + filename_sig + " and " + filename_ge)
        return render_template('predict.html', content_insert=' \n '.join('{}'.format(item) for item in predict(filename_sig, filename_ge)))

    return render_template("upload.html", content_insert=content_text)

def predict(filename_sig, filename_ge):
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
    model_parser.add_argument('--test_file', type=str, default=os.path.join(app.config['UPLOAD_FOLDER'], filename_sig))
    # Additional arguments
    model_parser.add_argument('--batch_size', type = int, default=64)
    model_parser.add_argument('--ae_input_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    model_parser.add_argument('--ae_label_file', type=str, default="/home/jrollins/home/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    model_parser.add_argument('--cell_ge_file', type=str, default=os.path.join(app.config['UPLOAD_FOLDER'], filename_ge), help='the file which used to map cell line to gene expression file')
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
    sorted_test_input = pd.read_csv(args.test_file).sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])

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
    model.load_state_dict(torch.load('/home/jrollins/home/MultiDCP/MultiDCP/website/MultiDCP_data/yoyo_07172022/1013rand1.pt', map_location = device))
    loc_predicted_result_for_testset = '/home/jrollins/home/MultiDCP/MultiDCP/website/results/' # Folder to save results
    results_name = 'results_df_' + datetime.now().strftime('%m%d%y%H%M') + '.csv'
    data_save = True

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
        if data_save:
            genes_cols = sorted_test_input.columns[5:]
            print(sorted_test_input.shape)
            print(predict_np.shape)
            #assert sorted_test_input.shape[0] == predict_np.shape[0] # ensure nothing gets filtered out, no testing data is out of research scope
            predict_df = pd.DataFrame(predict_np, columns = genes_cols)
            # hidden_df = pd.DataFrame(hidden_np, index = sorted_test_input.index, columns = [x for x in range(50)])
            ground_truth_df = pd.DataFrame(lb_np, columns = genes_cols)
            result_df  = pd.concat([sorted_test_input.iloc[:, :5], predict_df], axis = 1)
            ground_truth_df = pd.concat([sorted_test_input.iloc[:,:5], ground_truth_df], axis = 1)
            # hidden_df = pd.concat([sorted_test_input.iloc[:,:5], hidden_df], axis = 1) 
                    
            print("=====================================write out data=====================================")
            result_df.loc[[x for x in range(len(result_df))],:].to_csv(loc_predicted_result_for_testset + results_name, index = False)
            # hidden_df.loc[[x for x in range(len(hidden_df))],:].to_csv(hidden_repr_result_for_testset, index = False)
            # ground_truth_df.loc[[x for x in range(len(result_df))],:].to_csv('../MultiDCP/data/side_effect/test_for_same.csv', index = False)

    return result_df.head(20)

@app.route("/contact")
def contact():
    return render_template('contact.html')
