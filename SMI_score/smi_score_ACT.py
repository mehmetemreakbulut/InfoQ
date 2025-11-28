import os
import sys
import pandas as pd
import torch

from models.models import create_model
from SMI_score.visualization_utils import get_real_columns
from SMI_score.plot_inf_plane import plot_inf_plane

from SMI_score.score_utils import get_param_constraint, get_smi_scores_ACT, total_params, checkpoint_sizes, activation_sizes
import matplotlib.pyplot as plt

def get_last_run_folder(model_name, dataset):
    runs_path = os.path.join('RUNS', 'ACT', model_name, dataset)
    if not os.path.isdir(runs_path):
        raise FileNotFoundError(f"Path not found: {runs_path}")
    folders = [f for f in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, f))]
    if not folders:
        raise FileNotFoundError(f"No folders found in {runs_path}")
    last_folder = sorted(folders)[-1]
    return last_folder

def smi_score_act(model_name, dataset, BIT_WIDTHS=[2, 3, 4, 5, 6, 7, 8], folder_name=None):
    last_folder = get_last_run_folder(model_name, dataset)
    print(f"Last folder: {last_folder}")

    run_folder_path = os.path.join('RUNS', 'ACT', model_name, dataset, last_folder)
    results_csv_path = os.path.join(run_folder_path, 'results.csv')

    if not os.path.isfile(results_csv_path):
        raise FileNotFoundError(f"results.csv not found in {run_folder_path}")

    df = pd.read_csv(results_csv_path)
    # You can implement your functions with df below

    print(len(df.columns))

    #print df to table.txt in table format
    with open('table.txt', 'w') as f:
        f.write(df.to_string(index=False))

    if model_name == 'mobilenetv2':
        model = create_model('mobilenetv2_ALT', dataset)
    else:
        model = create_model(model_name, dataset)

    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8
   
    clms = get_real_columns(model_name, dataset)
    print(f"Columns in the DataFrame: {clms}")
    ly_corr, xl_corr = plot_inf_plane(df.copy(),clms.copy() , quant_dict_base.copy(), model_name, dataset)

    scores_smi = get_smi_scores_ACT(
        layer_bit_dict = quant_dict_base,
        column_names = df.columns.tolist(),
        score_df = df,
        XL_CORR=xl_corr,
        LY_CORR=ly_corr,
        model_name=model_name,
        dataset=dataset,
        BIT_WIDTHS = BIT_WIDTHS
    )
    found_layers_params = total_params(model, quant_dict_base.keys())
    checkpoint_sizes(list(found_layers_params.values()))
    param_constraint = activation_sizes(
        model= model,
        layer_bit_dict = quant_dict_base,
        BIT_WIDTHS=[2, 3, 4, 5, 6, 7, 8]
    )

    return scores_smi, param_constraint, quant_dict_base
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plots.py <model_name> <dataset>")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset = sys.argv[2]
    smi_score(model_name, dataset)



