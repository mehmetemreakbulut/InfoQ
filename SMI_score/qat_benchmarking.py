from SMI.dataloaders import get_dataloader, load_imagenet_256x256_validation
from SMI.utils.general_functions import evaluate_model
from SMI_score.other_algorithms.algorithm_scores import get_hawqv2_sensitivities, get_mpqco_sensitivities, get_ompq_sensitivities, get_limpq_sensitivities
from SMI_score.score_utils import checkpoint_sizes, find_top_n_configs, total_params
from SMI_score.smi_score import smi_score
from models.models import create_model
import copy
import torch # type: ignore
from tqdm import tqdm

from mqbench_quantization.quant import mpq_quantize, uniform_quantized_models

def train_model(model, train_dataloader, num_epochs, optimizer, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = batch['image'], batch['label']
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
    return model
def prepare_models(modelname, dataset, sizes, bitwidths = [2, 4, 8]):
    
    
    model = create_model(modelname, dataset)
    _, test_loader , train_dataloader = get_dataloader(dataset,batch_size=64, num_workers=16, subset=False)

    finetune_dataloader_subset = load_imagenet_256x256_validation()
    
    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8

    
    uniform_models = uniform_quantized_models(
        bitwidths= bitwidths.copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=16,
        calib_dataloader=train_dataloader,
        per_channel=False,
        activation_bit=8,
    )
    
    hawq_sensitivities, _ = get_hawqv2_sensitivities(
        eight_bit_model=uniform_models[8],
        four_bit_model=uniform_models[4],
        two_bit_model=uniform_models[2],
        quant_dict_base= quant_dict_base,
        modelname=modelname,
        dataset=dataset,
        BATCH_START=0,
        BATCH_END=80
    )

    mqpco_sensitivities, _ = get_mpqco_sensitivities(
        quant_dict_base=quant_dict_base,
        modelname=modelname,
        dataset=dataset,
        BATCH_START=0,
        BATCH_END=80
    )
    '''
    ompq_sensitivities = get_ompq_sensitivities(
        model = copy.deepcopy(model),
        modelname=modelname,
        dataset=dataset,
        quant_dict_base=quant_dict_base,
        bitwidths=bitwidths,
        base_bit=8
    )'''
    
   
   
  
  
    '''
    limpq_sensitivities = get_limpq_sensitivities(modelname)
    print(quant_dict_base)
    
    limpq_sensitivity_2 = {}
    limpq_sensitivity_2[0] = {2:-1e8,4:-1e8, 8:0}
    for key, value in limpq_sensitivities.items():
        limpq_sensitivity_2[key+1] = value
    limpq_sensitivities = limpq_sensitivity_2
    print(limpq_sensitivities)
    '''

    smi_sensitivities, param_constraint, _ = smi_score(modelname, dataset)

    smi_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    hawq_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    mqpco_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    #ompq_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}

    ompq_sensitivities = copy.deepcopy(smi_sensitivities)
    limpq_sensitivities = copy.deepcopy(mqpco_sensitivities)


    BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE = {}    
    BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_OMPQ_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_LIMPQ_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_INF_SCORE = {}
    best_allocations_list = [BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE, BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE, BEST_5_BIT_ASSIGNMENT_OMPQ_SCORE, BEST_5_BIT_ASSIGNMENT_LIMPQ_SCORE, BEST_5_BIT_ASSIGNMENT_INF_SCORE]
    for size in sizes:
        for i, s in enumerate([hawq_sensitivities, mqpco_sensitivities, ompq_sensitivities,limpq_sensitivities, smi_sensitivities]):
            top_configs = find_top_n_configs(
                1,
                quant_dict_base,
                s,
                param_constraint,
                max_param_size=size,
            )
            best_allocations_list[i][size] = []
            for row in top_configs.iterrows():
                assignment = row[1]['bit_assignment']
                best_allocations_list[i][size].append((assignment, row[1]['param_size']))


    
    # Each size, train model with one epoch, then evaluate
    hawq_accuracy = {}
    mpqco_accuracy = {}
    ompq_accuracy = {}
    limpq_accuracy = {}
    smi_accuracy = {}
    accuracy_list = [hawq_accuracy, mpqco_accuracy, ompq_accuracy, limpq_accuracy, smi_accuracy]
    for size in sizes:
        for i, s in enumerate(accuracy_list):
            if i == 2 or i == 3:
                # OMPQ and LIMPQ are not implemented yet
                continue

            accuracy_list[i][size] = [] 
            for row in best_allocations_list[i][size]:
                assignment = row[0]
                copy_quant_dict = copy.deepcopy(quant_dict_base)
                for j, b in enumerate(copy_quant_dict.keys()):
                    copy_quant_dict[b] = assignment[j]
                model_q = mpq_quantize(
                        layer_bitwidth_dict=copy_quant_dict,
                        num_calibration_batches=5,
                        calib_dataloader=train_dataloader,
                        bit_model_dict=uniform_models,
                )
                # Train the model for one epoch
                accuracies = []
                for _ in range(25):
                    copy_model = copy.deepcopy(model_q)
                    copy_model = train_model(
                        model=copy_model,
                        train_dataloader=finetune_dataloader_subset,
                        num_epochs=1,
                        optimizer=torch.optim.Adam(model_q.parameters(), lr=0.001),
                        device='cuda',
                    )
                    # Evaluate the model
                    accuracy = evaluate_model(copy_model, test_loader)
                    print('Size:', row[1], 'Accuracy:', accuracy)
                    accuracies.append(accuracy)
                
                accuracy_list[i][size].append({
                    'assignment': assignment,
                    'param_size': row[1],
                    'accuracies': accuracies,
                })


    #save the accuracy_list
    with open(f'SMI_score/evals/QAT/{modelname}_{dataset}_qat_accuracy_list.pkl', 'wb') as f:
        import pickle
        pickle.dump(accuracy_list, f)


def plot_results(modelname, dataset, sizes):
    import matplotlib.pyplot as plt
    import pickle

    with open(f'SMI_score/evals/QAT/{modelname}_{dataset}_qat_accuracy_list.pkl', 'rb') as f:
        accuracy_list = pickle.load(f)

    #x axis is sizes, plot each algorithm's average accuracy for each size and add error bars
    hawq_avg_accuracies = []
    hawq_std_accuracies = []
    mpqco_avg_accuracies = []
    mpqco_std_accuracies = []
    '''
    ompq_avg_accuracies = []
    ompq_std_accuracies = []
         '''
    limpq_avg_accuracies = []
    limpq_std_accuracies = []
    
    smi_avg_accuracies = []
    smi_std_accuracies = []

    for size in sizes:
        hawq_accuracies = [hawq['accuracies']  for hawq in accuracy_list[0][size]][0]
        print(len(hawq_accuracies))
        hawq_avg_accuracies.append(sum(hawq_accuracies) / len(hawq_accuracies))
        hawq_std_accuracies.append((max(hawq_accuracies) - min(hawq_accuracies)) / 2)

        mpqco_accuracies = [mpqco['accuracies'] for mpqco in accuracy_list[1][size]][0]
        mpqco_avg_accuracies.append(sum(mpqco_accuracies) / len(mpqco_accuracies))
        mpqco_std_accuracies.append((max(mpqco_accuracies) - min(mpqco_accuracies)) / 2)
        '''
        ompq_accuracies = [ompq['accuracies'] for ompq in accuracy_list[2][size]][0]
        ompq_avg_accuracies.append(sum(ompq_accuracies) / len(ompq_accuracies))
        ompq_std_accuracies.append((max(ompq_accuracies) - min(ompq_accuracies)) / 2)
             '''
        limpq_accuracies = [limpq['accuracies'] for limpq in accuracy_list[3][size]][0]
        limpq_avg_accuracies.append(sum(limpq_accuracies) / len(limpq_accuracies))
        limpq_std_accuracies.append((max(limpq_accuracies) - min(limpq_accuracies)) / 2)
    
        smi_accuracies = [smi['accuracies']for smi in accuracy_list[4][size]][0]
        smi_avg_accuracies.append(sum(smi_accuracies) / len(smi_accuracies))
        smi_std_accuracies.append((max(smi_accuracies) - min(smi_accuracies)) / 2)
    # Plotting
    plt.figure(figsize=(12, 9))
    #scale them to percentage
    hawq_avg_accuracies = [acc * 100 for acc in hawq_avg_accuracies]
    hawq_std_accuracies = [std * 100 for std in hawq_std_accuracies]
    mpqco_avg_accuracies = [acc * 100 for acc in mpqco_avg_accuracies]
    mpqco_std_accuracies = [std * 100 for std in mpqco_std_accuracies]
    '''
    ompq_avg_accuracies = [acc * 100 for acc in ompq_avg_accuracies]
    ompq_std_accuracies = [std * 100 for std in ompq_std_accuracies]
        '''
    limpq_avg_accuracies = [acc * 100 for acc in limpq_avg_accuracies]
    limpq_std_accuracies = [std * 100 for std in limpq_std_accuracies]

    smi_avg_accuracies = [acc * 100 for acc in smi_avg_accuracies]
    smi_std_accuracies = [std * 100 for std in smi_std_accuracies]
    plt.errorbar(sizes, hawq_avg_accuracies, yerr=hawq_std_accuracies, label='HAWQv2', fmt='-o', capsize=7, color='green', linewidth=4, markersize=8, capthick=3)
    plt.errorbar(sizes, mpqco_avg_accuracies, yerr=mpqco_std_accuracies, label='MPQCO', fmt='-o', capsize=7, color='red', linewidth=4, markersize=8, capthick=3)
   
    #plt.errorbar(sizes, ompq_avg_accuracies, yerr=ompq_std_accuracies, label='OMPQ', fmt='-o', capsize=5, color='purple')
    plt.errorbar(sizes, limpq_avg_accuracies, yerr=limpq_std_accuracies, label='LIMPQ', fmt='-o', capsize=7, color='orange', linewidth=4, markersize=8, capthick=3)
    plt.errorbar(sizes, smi_avg_accuracies, yerr=smi_std_accuracies, label='Ours', fmt='-o', capsize=7, color='blue', linewidth=4,  markersize=8, capthick=3)
    plt.title(f'ResNet18 QAT Results', fontsize=54) 
    plt.xlabel('Model Size (MB)', fontsize=54)
    plt.ylabel('Accuracy (%)', fontsize=54)
    plt.yticks(fontsize=44)
    plt.xticks(sizes, fontsize=44)
    #make y_ticks with precision of 1
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    #set legend right of the plot
    plt.legend(fontsize=36, title_fontsize=18, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'SMI_score/evals/QAT/{modelname}_{dataset}_qat_accuracy_plot.png')
    plt.show()


import sys
import os
if __name__ == "__main__":
    model = 'resnet18_gluon'
    dataset = 'imagenet_256x256'
    sizes = [3.8, 4.0, 4.2, 4.4]
    print(sizes)

    plot_results(model, dataset, sizes)
    exit()

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging to redirect print statements to file
    log_filename = f'logs/qat_benchmarking_{model}_{dataset}.log'

    
    # Create a custom print function that writes to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # Ensure immediate writing
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    # Redirect stdout to our logger
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"Logging output to: {log_filename}")
    print(f"Starting benchmarking for {model} on {dataset}")
    print("=" * 50) 
    
    prepare_models(
        modelname=model,
        dataset=dataset,
        sizes=sizes,
        bitwidths=[2, 4, 8]
    )