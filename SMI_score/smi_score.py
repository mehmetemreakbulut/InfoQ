import os
import sys
import pandas as pd
import torch

from models.models import create_model
from SMI_score.visualization_utils import get_real_columns
from SMI_score.plot_inf_plane import plot_inf_plane

from SMI_score.score_utils import get_param_constraint, get_smi_scores, total_params, checkpoint_sizes
import matplotlib.pyplot as plt

def get_last_run_folder(model_name, dataset):
    runs_path = os.path.join('RUNS', model_name, dataset)
    if not os.path.isdir(runs_path):
        raise FileNotFoundError(f"Path not found: {runs_path}")
    folders = [f for f in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, f))]
    if not folders:
        raise FileNotFoundError(f"No folders found in {runs_path}")
    last_folder = sorted(folders)[-1]
    return last_folder

def smi_score(model_name, dataset, BIT_WIDTHS=[2, 4, 8], folder_name=None):
    if folder_name is None:
        last_folder = get_last_run_folder(model_name, dataset)
    else:
        last_folder = folder_name
    print(f"Last folder: {last_folder}")

    run_folder_path = os.path.join('RUNS', model_name, dataset, last_folder)
    results_csv_path = os.path.join(run_folder_path, 'results.csv')

    if not os.path.isfile(results_csv_path):
        raise FileNotFoundError(f"results.csv not found in {run_folder_path}")

    df = pd.read_csv(results_csv_path)
    # You can implement your functions with df below


    if model_name == 'mobilenetv2':
        model = create_model('mobilenetv2_ALT', dataset)
    else:
        model = create_model(model_name, dataset)
    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8
   

    ly_corr, xl_corr = plot_inf_plane(df.copy(), get_real_columns(model_name, dataset).copy(), quant_dict_base.copy(), model_name, dataset)

    scores_smi = get_smi_scores(
        layer_bit_dict = quant_dict_base,
        column_names = df.columns.tolist(),
        score_df = df,
        XL_CORR=xl_corr,
        LY_CORR=ly_corr,
        model_name=model_name,
        dataset=dataset,
        BIT_WIDTHS = BIT_WIDTHS
    )
    figure = plt.figure(figsize=(10, 5))
    two_bit_scores = []
    four_bit_scores = []
    eight_bit_scores = []
    if 3 in BIT_WIDTHS:
        three_bit_scores = []
    if 5 in BIT_WIDTHS:
        five_bit_scores = []
    if 6 in BIT_WIDTHS:
        six_bit_scores = []
    if 7 in BIT_WIDTHS:
        seven_bit_scores = []
    for i in range(len(scores_smi)):
        two_bit_scores.append(scores_smi[i][2])
        four_bit_scores.append(scores_smi[i][4])
        eight_bit_scores.append(scores_smi[i][8])
        if len(BIT_WIDTHS) > 3:
            if 3 in BIT_WIDTHS:
                three_bit_scores.append(scores_smi[i][3])
            if 5 in BIT_WIDTHS:
                five_bit_scores.append(scores_smi[i][5])
            if 6 in BIT_WIDTHS:
                six_bit_scores.append(scores_smi[i][6])
            if 7 in BIT_WIDTHS:
                seven_bit_scores.append(scores_smi[i][7])
   
    plt.plot(two_bit_scores, label='2-bit') 
    plt.plot(four_bit_scores, label='4-bit')
    plt.plot(eight_bit_scores, label='8-bit')
    if len(BIT_WIDTHS) > 3:
        if 3 in BIT_WIDTHS:
            plt.plot(three_bit_scores, label='3-bit')
        if 5 in BIT_WIDTHS:
            plt.plot(five_bit_scores, label='5-bit')
        if 6 in BIT_WIDTHS:
            plt.plot(six_bit_scores, label='6-bit')
        if 7 in BIT_WIDTHS:
            plt.plot(seven_bit_scores, label='7-bit')   
    
    plt.xlabel('Layer Index')
    plt.xticks(range(len(scores_smi)), [f'{i}' for i in range(len(scores_smi))], rotation=0)
    plt.ylabel('SMI Score')
    plt.title('SMI Scores for Different Bit Widths')
    plt.legend()
    plt.tight_layout()
    os.makedirs(f'SMI_score/{dataset}/{model_name}', exist_ok=True)
    plt.savefig(f'SMI_score/{dataset}/{model_name}/smi_scores_plot.png')
    plt.show()

    found_layers_params = total_params(model, quant_dict_base.keys())
    checkpoint_sizes(list(found_layers_params.values()))
    param_constraint = get_param_constraint(
        layer_bit_dict = quant_dict_base,
        params = list(found_layers_params.values()),
        BIT_WIDTHS = BIT_WIDTHS
    )

    return scores_smi, param_constraint, quant_dict_base
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plots.py <model_name> <dataset>")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset = sys.argv[2]
    smi_score(model_name, dataset)



