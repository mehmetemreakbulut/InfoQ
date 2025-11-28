from SMI.dataloaders import get_dataloader
from SMI.utils.general_functions import evaluate_model
from SMI_score.other_algorithms.algorithm_scores import get_hawqv2_sensitivities, get_mpqco_sensitivities, get_ompq_sensitivities, get_limpq_sensitivities
from SMI_score.score_utils import checkpoint_sizes, find_top_n_configs, total_params
from SMI_score.smi_score import smi_score
from models.models import create_model
import copy
import torch # type: ignore

from mqbench_quantization.quant import mpq_quantize, uniform_quantized_models
#from mqbench_quantization.quant_old import quantize_ptq_mixed_precision

def benchmark(modelname, dataset, sizes, bitwidths = [2, 4, 8]):
    
    model = create_model(modelname, dataset)

    print(f"Benchmarking {modelname} on {dataset} with sizes {sizes}")


    _, test_loader , train_dataloader = get_dataloader(dataset,batch_size=64, num_workers=16, subset=False)

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
    )
    '''
    
   
   
  
  
    
    limpq_sensitivities = get_limpq_sensitivities(modelname)
    print(quant_dict_base)
    
    limpq_sensitivity_2 = {}
    limpq_sensitivity_2[0] = {2:-1e8,4:-1e8, 8:0}
    for key, value in limpq_sensitivities.items():
        limpq_sensitivity_2[key+1] = value
    limpq_sensitivities = limpq_sensitivity_2
    print(limpq_sensitivities)
    
    

 
    smi_sensitivities, param_constraint, _ = smi_score(modelname, dataset)

    #SET FIRST LAYERS TO 8 as an heuristic:
    smi_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    hawq_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    mqpco_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    #ompq_sensitivities[0] = {2:-1e8,4:-1e8, 8:0}
    #ompq_sensitivities = { i: {2:-1e8,4:-1e4, 8:0} for i in range(0, len(quant_dict_base))}
    #limpq_sensitivities = { i: {2:-1e8,4:-1e4, 8:0} for i in range(0, len(quant_dict_base))}
    

    #PRINT LENGTH OF SENSITIVITIES
    print(f"Hawq sensitivities length: {len(hawq_sensitivities)}")
    print(f"MPQCO sensitivities length: {len(mqpco_sensitivities)}")
    #print(f"OMPQ sensitivities length: {len(ompq_sensitivities)}")  
    print(f"LIMPQ sensitivities length: {len(limpq_sensitivities)}")  
    print(f"SMI sensitivities length: {len(smi_sensitivities)}")
    #quant dict base length
    print(f"Quant dict base length: {len(quant_dict_base)}")

    #print(smi_sensitivities)
    if not (len(hawq_sensitivities) == len(mqpco_sensitivities) == len(ompq_sensitivities) == len(limpq_sensitivities) == len(smi_sensitivities) == len(quant_dict_base)):
        raise ValueError("Sensitivity lengths are not equal to quant_dict_base length. Please check the sensitivities.")

    BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE = {}    
    BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE = {}
    #BEST_5_BIT_ASSIGNMENT_OMPQ_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_LIMPQ_SCORE = {}
    BEST_5_BIT_ASSIGNMENT_INF_SCORE = {}
    best_allocations_list = [BEST_5_BIT_ASSIGNMENT_HAWQ_SCORE, BEST_5_BIT_ASSIGNMENT_MPQCO_SCORE, BEST_5_BIT_ASSIGNMENT_LIMPQ_SCORE, BEST_5_BIT_ASSIGNMENT_INF_SCORE]
    for size in sizes:
        for i, s in enumerate([hawq_sensitivities, mqpco_sensitivities,limpq_sensitivities, smi_sensitivities]):
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

    hawq_accuracy = {}
    mpqco_accuracy = {}
    limpq_accuracy = {}
    smi_accuracy = {}
    accuracy_list = [hawq_accuracy, mpqco_accuracy, limpq_accuracy, smi_accuracy]
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
    _, test_loader , train_dataloader = get_dataloader(dataset,batch_size=32, num_workers=32, subset=False)
    #train_dataloader, test_loader, _, _ = get_dataloader(dataset, batch_size=256, num_workers=16)
    uniform_models = uniform_quantized_models(
        bitwidths= bitwidths.copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=16,
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

def plot_agg(hawq_score_acc,mpqco_score_acc,ompq_score_acc,limpq_score_acc, inf_score_acc,modelname, dataset, sizes, uniform_model_dict, avg=True):
    agg_accuracies = {
        'Ours': [],
        'Hawqv2': [],
        'MPQCO': [],
        'OMPQ': [],
        'LIMPQ': [],
        'UBQ': [],
        }
    for size in  sizes:
        def get_agg_accuracy(results):
            if not results:
                return np.nan
            if avg:
                return np.mean([r['accuracy'] for r in results])
            else:
                # If not averaging, return the maximum accuracy for the size
                return max([r['accuracy'] for r in results])

        # Populate the avg_accuracies dictionary.
        # .get(size, []) provides an empty list if a size key is missing for an algorithm.
        agg_accuracies['Ours'].append(get_agg_accuracy(inf_score_acc.get(size, [])))
        agg_accuracies['Hawqv2'].append(get_agg_accuracy(hawq_score_acc.get(size, [])))
        agg_accuracies['MPQCO'].append(get_agg_accuracy(mpqco_score_acc.get(size, [])))
        agg_accuracies['OMPQ'].append(get_agg_accuracy(ompq_score_acc.get(size, [])))
        agg_accuracies['LIMPQ'].append(get_agg_accuracy(limpq_score_acc.get(size, [])))


    # --- Step 2: Create the plot ---
    plt.figure(figsize=(15, 10))

    # Plot average accuracy vs. size, connecting the points for each algorithm.
    # The 'marker' argument creates the scatter point, and 'linestyle' connects them.
    plt.plot(sizes, agg_accuracies['Ours'], color='blue', label='Ours', marker='o', linestyle='-', markersize=12, linewidth=4)
    plt.plot(sizes, agg_accuracies['Hawqv2'], color='green', label='Hawqv2', marker='s', linestyle='-', markersize=12, linewidth=4)
    plt.plot(sizes, agg_accuracies['MPQCO'], color='red', label='MPQCO', marker='^', linestyle='-', markersize=12, linewidth=4)
    plt.plot(sizes, agg_accuracies['OMPQ'], color='purple', label='OMPQ', marker='*', linestyle='-', markersize=12, linewidth=4)
    plt.plot(sizes, agg_accuracies['LIMPQ'], color='orange', label='LIMPQ', marker='x', linestyle='-', markersize=12, linewidth=4)

    size_ubq = []
    for size, acc in uniform_model_dict.items():
        agg_accuracies['UBQ'].append(acc['accuracy'])
        size_ubq.append(acc['param_size'])
    plt.plot(size_ubq, agg_accuracies['UBQ'], color='black', label='UBQ', marker='x', linestyle='-')
    #annotate the UBQ points with their bitwidths
    for i, bit in enumerate(uniform_model_dict.keys()):
        plt.annotate(f'{bit}b', (size_ubq[i], agg_accuracies['UBQ'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black')
    # --- Step 3: Configure plot labels, title, and appearance ---
    plt.xlabel('Parameter Size Constraint (MB)', fontsize=14)
    if avg:
        plt.ylabel('Average Accuracy (%)', fontsize=14)
        plt.title(f'Average Accuracy vs. Parameter Size Constraint for {modelname} on {dataset}', fontsize=16)
    else:
        plt.ylabel('Maximum Accuracy (%)', fontsize=14)
        plt.title(f'Maximum Accuracy vs. Parameter Size Constraint for {modelname} on {dataset}', fontsize=16)

    # Set x-ticks to correspond to the tested sizes for clarity.
    #sizes.append(size_ubq[-1])  # Ensure the last size is included for UBQ

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
    if avg:
        output_filename = f'SMI_score/evals/{modelname}_{dataset}_avg_accuracy_vs_size.png'
    else:
        output_filename = f'SMI_score/evals/{modelname}_{dataset}_max_accuracy_vs_size.png'
    # Uncomment the next line to save the figure
    plt.savefig(output_filename, dpi=300)
    print(f"Plot generated. To save, uncomment the plt.savefig line. Filename: {output_filename}")

    plt.show()


import pickle
#from SMI_score.loader_benchmark import loader_highres_train
def test():

    model = create_model('resnet18_gluon', 'imagenet_256x256') # resnet18(weights="IMAGENET1K_V1")#

    #print(f"Benchmarking {modelname} on {dataset} with sizes {sizes}")


    train_dataloader, test_loader , _= get_dataloader('imagenet_256x256',batch_size=64, num_workers=32, subset=False)
  
    print('BASE MODEL', evaluate_model(model, test_loader))
    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8
    

    uniform_models = uniform_quantized_models(
        bitwidths= [2,4,8].copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=1,
        calib_dataloader=train_dataloader,
        per_channel=False,
        activation_bit=8,
    )
    copy8 = copy.deepcopy(uniform_models[8])

    accuracy = evaluate_model(copy8, test_loader)
    print('Eightbit ACC:', accuracy)

    assignment = [8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 2]
    copy_quant_dict = copy.deepcopy(quant_dict_base)
    for j, b in enumerate(copy_quant_dict.keys()):
        copy_quant_dict[b] = assignment[j]
    model_q = mpq_quantize(
            layer_bitwidth_dict=copy_quant_dict,
            num_calibration_batches=10,
            calib_dataloader=train_dataloader,
            bit_model_dict=uniform_models,
    )
    with open('test.txt', 'w') as f:
    # Print the model summary, redirecting the output to the file object 'f'
        print(model_q, file=f)
    accuracy = evaluate_model(model_q, test_loader)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":

    

    #test()
    #exit()
    modelname ='resnet18_gluon'
    dataset = 'imagenet_256x256'
    #Write the print statements in this file to a file called logs/benchmarking_imagenet_{modelname}_{dataset}.txt
    import os
    import sys
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging to redirect print statements to file
    log_filename = f'logs/benchmarking_imagenet_{modelname}_{dataset}.txt'
    
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
    print(f"Starting benchmarking for {modelname} on {dataset}")
    print("=" * 50) 
    #sizes = [3.6, 3.8, 4, 4.1 ,4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8] #RESNET18 real
    sizes = [4, 4.1, 4.2, 4.3, 4.4, 4.5,  5,  5.5, 6, 6.5, 7, 7.5, 8] # RESNET18 real

    #sizes = [10.5, 10.75, 11,11.25,11.5,11.75,12, 12.25, 12.50, 12.75, 13, 14, 17, 20, 22.4] #resnet50
    #sizes = [10.5, 11, 11.5, 12, 13, 14, 17,  20, 22.4]


    #sizes = [0.78, 0.79, 0.80, 0.96, 1.10, 1.34, 1.58, 1.82, 2.06] #mobilenetv2


    #sizes = [7.2, 7.5, 7.8, 8.1, 8.4, 8.7, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.2, 17.7, 20.3]
    #sizes = [7.2, 7.8, 8.4, 8.7, 10.5, 12.5, 14.5, 15.2, 17.7, 20.3]
    benchmark(modelname, dataset, sizes)
    exit()
    #uniform_models_evaluation(modelname, dataset, [3,4,5,6])
    #exit()
    #open SMI_score/evals/{modelname}_{dataset}_accuracy.pkl
    with open(f'SMI_score/evals/{modelname}_{dataset}_accuracy.pkl', 'rb') as f:
        accuracy_list = pickle.load(f)

    with open(f'SMI_score/evals/{modelname}_{dataset}_uniform_models.pkl', 'rb') as f:
        uniform_model_dict = pickle.load(f)
    

    del uniform_model_dict[8]


    plot_agg(accuracy_list[0], accuracy_list[1], accuracy_list[2], accuracy_list[3],accuracy_list[4], modelname, dataset, sizes, uniform_model_dict, avg=True)
    plot_agg(accuracy_list[0], accuracy_list[1], accuracy_list[2], accuracy_list[3],accuracy_list[4], modelname, dataset, sizes, uniform_model_dict, avg=False)
  

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
    hawq_score_acc, mpqco_score_acc, ompq_score_acc, limpq_score_acc, inf_score_acc = accuracy_list

    for size in  sizes:
        inf_score_results = inf_score_acc[size]
        hawq_score_results = hawq_score_acc[size]
        mpqco_score_results = mpqco_score_acc[size]
        #ompq_results = ompq_score_acc[size]
        #limpq_results = limpq_score_acc[size]
        #print allocation of ompq_score_results
        #inf_score_results = [max(inf_score_results, key=lambda x: x['accuracy'])]
        #hawq_score_results = [max(hawq_score_results, key=lambda x: x['accuracy'])]
        #mpqco_score_results = [max(mpqco_score_results, key=lambda x: x['accuracy'])]
        #ompq_results = [max(ompq_results, key=lambda x: x['accuracy'])]
        #limpq_results = [max(limpq_results, key=lambda x: x['accuracy'])]
        #scatter plot, y axis is accuracy, x axis is param size
        inf_score_x = [result['param_size'] for result in inf_score_results]
        inf_score_y = [result['accuracy'] for result in inf_score_results]

        hawq_score_x = [result['param_size'] for result in hawq_score_results]
        hawq_score_y = [result['accuracy'] for result in hawq_score_results]

        mpqco_score_x = [result['param_size'] for result in mpqco_score_results]
        mpqco_score_y = [result['accuracy'] for result in mpqco_score_results]

        #ompq_score_x = [result['param_size'] for result in ompq_results]
        #ompq_score_y = [result['accuracy'] for result in ompq_results]

        #limpq_score_x = [result['param_size'] for result in limpq_results]
        #limpq_score_y = [result['accuracy'] for result in limpq_results]


        #scatter plot, use different shape and color for each method, one with triangle other with circle
        

        if size == sizes[0]:
            plt.scatter(inf_score_x, inf_score_y, color='blue', label='Ours', marker='o', linewidth=6)
            plt.scatter(hawq_score_x, hawq_score_y, color='green', label='Hawqv2', marker='s', linewidth=6)
            plt.scatter(mpqco_score_x, mpqco_score_y, color='red', label='MPQCO', marker='^', linewidth=6)
            #plt.scatter(ompq_score_x, ompq_score_y, color='purple', label='OMPQ', marker='*', linewidth=6)
            #plt.scatter(limpq_score_x, limpq_score_y, color='orange', label='OMPQ', marker='x', linewidth=6)
        else:
            plt.scatter(inf_score_x, inf_score_y, color='blue',  marker='o', linewidth=6)
            plt.scatter(hawq_score_x, hawq_score_y, color='green',  marker='s', linewidth=6)
            plt.scatter(mpqco_score_x, mpqco_score_y, color='red',  marker='^', linewidth=6)
            #plt.scatter(ompq_score_x, ompq_score_y, color='purple',  marker='*', linewidth=6)
            #plt.scatter(limpq_score_x, limpq_score_y, color='orange',  marker='x', linewidth=6)


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
    
    # Close the log file and restore stdout
    logger.close()
    sys.stdout = logger.terminal
    print(f"Logging completed. Output saved to: {log_filename}")


