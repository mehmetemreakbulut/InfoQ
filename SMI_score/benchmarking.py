from SMI.dataloaders import get_dataloader
from SMI.utils.general_functions import evaluate_model
from SMI_score.other_algorithms.algorithm_scores import get_hawqv2_sensitivities, get_mpqco_sensitivities, get_ompq_sensitivities
from SMI_score.score_utils import checkpoint_sizes, find_top_n_configs, total_params
from SMI_score.smi_score import smi_score
from models.models import create_model
import copy
import torch

from mqbench_quantization.quant import mpq_quantize, uniform_quantized_models

def benchmark(modelname, dataset, sizes, bitwidths = [2, 4, 8]):
    
    model = create_model(modelname, dataset)

    print(f"Benchmarking {modelname} on {dataset} with sizes {sizes}")


    train_dataloader, test_loader , _ , _= get_dataloader(dataset, batch_size=64)

    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8




    
    uniform_models = uniform_quantized_models(
        bitwidths= bitwidths.copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=5,
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
    )

    mqpco_sensitivities, _ = get_mpqco_sensitivities(
        quant_dict_base=quant_dict_base,
        modelname=modelname,
        dataset=dataset,
    )

    ompq_sensitivities = get_ompq_sensitivities(
        model = copy.deepcopy(model),
        modelname=modelname,
        dataset=dataset,
        quant_dict_base=quant_dict_base,
        bitwidths=bitwidths,
        base_bit=8
    )


    smi_sensitivities, param_constraint, _ = smi_score(modelname, dataset)

    #smi_sensitivities[0] = {2:-1000,4:-1000, 8:0}

    #PRINT LENGTH OF SENSITIVITIES
    print(f"Hawq sensitivities length: {len(hawq_sensitivities)}")
    print(f"MPQCO sensitivities length: {len(mqpco_sensitivities)}")
    print(f"OMPQ sensitivities length: {len(ompq_sensitivities)}")  
    print(f"SMI sensitivities length: {len(smi_sensitivities)}")
    #quant dict base length
    print(f"Quant dict base length: {len(quant_dict_base)}")

    #print(smi_sensitivities)
 

    BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE = {}    
    BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_OMPQ_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_INF_SCORE = {}
    best_allocations_list = [BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE, BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE, BEST_5_BIT_ASSIGNMENT_OMPQ_SCORE, BEST_5_BIT_ASSIGNMENT_INF_SCORE]
    for size in sizes:
        for i, s in enumerate([hawq_sensitivities, mqpco_sensitivities, ompq_sensitivities, smi_sensitivities]):
            top_configs = find_top_n_configs(
                3,
                quant_dict_base,
                s,
                param_constraint,
                max_param_size=size,
            )
            best_allocations_list[i][size] = []
            for row in top_configs.iterrows():
                assignment = row[1]['bit_assignment']
                best_allocations_list[i][size].append((assignment, row[1]['param_size']))

    hawq_accuracy = {}
    mpqco_accuracy = {}
    ompq_accuracy = {}
    smi_accuracy = {}
    accuracy_list = [hawq_accuracy, mpqco_accuracy, ompq_accuracy, smi_accuracy]
    for size in sizes:
        for i, s in enumerate(accuracy_list):
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
                accuracy = evaluate_model(model_q, test_loader)
                print(f"Size: {row[1]}, Accuracy: {accuracy}")
                accuracy_list[i][size].append({
                    'assignment': assignment,
                    'param_size': row[1],
                    'accuracy': accuracy
                })

    #save accuracy list with modelname and dataset
    import os
    os.makedirs(f'SMI_score/evals', exist_ok=True)
    with open(f'SMI_score/evals/{modelname}_{dataset}_accuracy.pkl', 'wb') as f:
        import pickle
        pickle.dump(accuracy_list, f)

    
def uniform_models_evaluation(model_name, dataset, bitwidths = [2, 4, 8]):

    model = create_model(model_name, dataset)
    train_dataloader, test_loader, _, _ = get_dataloader(dataset, batch_size=64)
    uniform_models = uniform_quantized_models(
        bitwidths= bitwidths.copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=5,
        calib_dataloader=train_dataloader,
        per_channel=False,
        activation_bit=8,
    )

    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8

    # Evaluate the uniform models
    found_layers_params = total_params(model, quant_dict_base.keys())
    checkpoint_size_dict = checkpoint_sizes(list(found_layers_params.values()))
    values = {}
    for bitwidth, model_q in uniform_models.items():
        accuracy = evaluate_model(model_q, test_loader)
        print(f"Bitwidth: {bitwidth}, Accuracy: {accuracy}")
        values[bitwidth] = {
            'accuracy': accuracy,
            'param_size': checkpoint_size_dict[str(bitwidth)],
        }
    #save the results
    import os
    os.makedirs(f'SMI_score/evals', exist_ok=True)
    with open(f'SMI_score/evals/{model_name}_{dataset}_uniform_models.pkl', 'wb') as f:
        import pickle
        pickle.dump(values, f)

import numpy as np
import matplotlib.pyplot as plt

def plot_avg(hawq_score_acc,mpqco_score_acc,ompq_score_acc,inf_score_acc,modelname, dataset, sizes, uniform_model_dict):
    avg_accuracies = {
        'Ours': [],
        'Hawqv2': [],
        'MPQCO': [],
        'OMPQ': [],
        'UBQ': [],
        }
    for size in  sizes:
        def get_avg_accuracy(results):
            if not results:
                return np.nan
            return np.mean([r['accuracy'] for r in results])

        # Populate the avg_accuracies dictionary.
        # .get(size, []) provides an empty list if a size key is missing for an algorithm.
        avg_accuracies['Ours'].append(get_avg_accuracy(inf_score_acc.get(size, [])))
        avg_accuracies['Hawqv2'].append(get_avg_accuracy(hawq_score_acc.get(size, [])))
        avg_accuracies['MPQCO'].append(get_avg_accuracy(mpqco_score_acc.get(size, [])))
        avg_accuracies['OMPQ'].append(get_avg_accuracy(ompq_score_acc.get(size, [])))


    # --- Step 2: Create the plot ---
    plt.figure(figsize=(15, 10))

    # Plot average accuracy vs. size, connecting the points for each algorithm.
    # The 'marker' argument creates the scatter point, and 'linestyle' connects them.
    plt.plot(sizes, avg_accuracies['Ours'], color='blue', label='Ours (Avg)', marker='o', linestyle='-')
    plt.plot(sizes, avg_accuracies['Hawqv2'], color='green', label='Hawqv2 (Avg)', marker='s', linestyle='-')
    plt.plot(sizes, avg_accuracies['MPQCO'], color='red', label='MPQCO (Avg)', marker='^', linestyle='-')
    plt.plot(sizes, avg_accuracies['OMPQ'], color='purple', label='OMPQ (Avg)', marker='*', linestyle='-')

    #plot uniform models in single model called 'UBQ'
    size_ubq = []
    for size, acc in uniform_model_dict.items():
        avg_accuracies['UBQ'].append(acc['accuracy'])
        size_ubq.append(acc['param_size'])
    plt.plot(size_ubq, avg_accuracies['UBQ'], color='black', label='UBQ', marker='x', linestyle='-')
    #annotate the UBQ points with their bitwidths
    for i, bit in enumerate(uniform_model_dict.keys()):
        plt.annotate(f'{bit}b', (size_ubq[i], avg_accuracies['UBQ'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black')
    # --- Step 3: Configure plot labels, title, and appearance ---
    plt.xlabel('Parameter Size Constraint (MB)', fontsize=14)
    plt.ylabel('Average Accuracy (%)', fontsize=14)
    plt.title(f'Average Accuracy vs. Parameter Size Constraint for {modelname} on {dataset}', fontsize=16)

    # Set x-ticks to correspond to the tested sizes for clarity.
    sizes.append(size_ubq[-1])  # Ensure the last size is included for UBQ

    plt.xticks(sizes, [f'{size:.3f}' for size in sizes])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation=90)

    # Add a legend to identify the algorithms.
    plt.legend(fontsize=12)

    # Add a grid with dashed lines for better readability.
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Ensure the layout is tight.
    plt.tight_layout()

    # --- Step 4: Save and show the plot ---
    output_filename = f'SMI_score/evals/{modelname}_{dataset}_avg_accuracy_vs_size.png'
    # Uncomment the next line to save the figure
    plt.savefig(output_filename, dpi=300)
    print(f"Plot generated. To save, uncomment the plt.savefig line. Filename: {output_filename}")

    plt.show()


import pickle
if __name__ == "__main__":
    modelname ='resnet20'
    dataset = 'cifar10'
    sizes = [0.080, 0.085, 0.090, 0.095, 0.1, 0.105, 0.110, 0.115, 0.12, 0.125, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24]  # Example sizes, adjust as needed
    #sizes = [0.165,0.175,0.185, 0.195, 0.205, 0.215, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.50]  # Example sizes, adjust as needed
    #benchmark(modelname, dataset, sizes)
    #uniform_models_evaluation(modelname, dataset, [2,3,4,8])
    #exit()
    #open SMI_score/evals/{modelname}_{dataset}_accuracy.pkl
    with open(f'SMI_score/evals/{modelname}_{dataset}_accuracy.pkl', 'rb') as f:
        accuracy_list = pickle.load(f)

    with open(f'SMI_score/evals/{modelname}_{dataset}_uniform_models.pkl', 'rb') as f:
        uniform_model_dict = pickle.load(f)


    plot_avg(accuracy_list[0], accuracy_list[1], accuracy_list[2], accuracy_list[3], modelname, dataset, sizes, uniform_model_dict)
    exit()

    #plot accuracies of different algorithms with respect to sizes, x axis are sizes, y axis are accuracies
    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns  
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    figure = plt.figure(figsize=(15, 10))
    hawq_score_acc, mpqco_score_acc, ompq_score_acc, inf_score_acc = accuracy_list
    print([a for a in hawq_score_acc])
    for size in  sizes:
        inf_score_results = inf_score_acc[size]
        hawq_score_results = hawq_score_acc[size]
        mpqco_score_results = mpqco_score_acc[size]
        #print allocation of ompq_score_results

        #scatter plot, y axis is accuracy, x axis is param size
        inf_score_x = [result['param_size'] for result in inf_score_results]
        inf_score_y = [result['accuracy'] for result in inf_score_results]

        hawq_score_x = [result['param_size'] for result in hawq_score_results]
        hawq_score_y = [result['accuracy'] for result in hawq_score_results]

        mpqco_score_x = [result['param_size'] for result in mpqco_score_results]
        mpqco_score_y = [result['accuracy'] for result in mpqco_score_results]

        ompq_score_x = [result['param_size'] for result in ompq_score_acc[size]]
        ompq_score_y = [result['accuracy'] for result in ompq_score_acc[size]]


        #scatter plot, use different shape and color for each method, one with triangle other with circle
        

        if size == sizes[0]:
            plt.scatter(inf_score_x, inf_score_y, color='blue', label='Ours', marker='o')
            plt.scatter(hawq_score_x, hawq_score_y, color='green', label='Hawqv2', marker='s')
            plt.scatter(mpqco_score_x, mpqco_score_y, color='red', label='MPQCO', marker='^')
            plt.scatter(ompq_score_x, ompq_score_y, color='purple', label='OMPQ', marker='*')
        else:
            plt.scatter(inf_score_x, inf_score_y, color='blue',  marker='o')
            plt.scatter(hawq_score_x, hawq_score_y, color='green',  marker='s')
            plt.scatter(mpqco_score_x, mpqco_score_y, color='red',  marker='^')
            plt.scatter(ompq_score_x, ompq_score_y, color='purple',  marker='*')


    plt.xlabel('Parameter Size(MB)')
    #make xticks 0.4, 0.6, 0.8, 1, 1.2, 1.4
    plt.xticks(sizes, [f'{size:.3f}' for size in sizes])

    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Parameter Size for Size Constraint')
    plt.legend()
    #add grid with dashed lines
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #save
    plt.savefig(f'SMI_score/evals/{modelname}_{dataset}_accuracy_vs_size.png')
    plt.show()


