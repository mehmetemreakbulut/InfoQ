#plot squeeze_net_smi_df, subplot for each layer, color with accuracies, shape of each layer is different
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def plot_inf_plane(df, columns, quant_dict_base, model_name, dataset):
    df.columns = columns
    columns.remove('Unnamed: 0')
    columns.remove('layer')
    columns.remove('bw')
    columns.remove('acc')
    #make tree columns
    plot_per_column = 4
    num_layers = len(columns)//2
    num_rows = (num_layers + plot_per_column - 1) // plot_per_column  # Ceiling division
    fig, axes = plt.subplots(num_rows, plot_per_column, figsize=(20, 5 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes]  # Flatten if multiple rows
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=df['acc'].min(), vmax=df['acc'].max())
    correlations_xl_acc = []
    correlations_ly_acc = []
    correlations_beta = {}
    betas = np.linspace(-10, 10, 1000)
    l_names = []
    def find_index_in_layers(block):
        i = 0
        for j,l in enumerate(list(quant_dict_base.keys())):
            if l.startswith(block):
                i = j
        return i
    for i, layer in enumerate(columns):
        if 'xl' in layer:
            continue
        ax = axes[i//2]
        #plot value versus accuracy
        
        #get where not NaN values

        subset = df[layer].notna() & df['acc'].notna()
        subset = df[subset]
    


        x = subset[layer[:-6]+'_mi_xl'].values
        y = subset[layer].values
        acc = subset['acc'].values



        #print correlation between x, y and acc
        print(f"Correlation between {layer} SMI(X,L) and accuracy: {np.corrcoef(x, acc)[0, 1]}")
        correlations_xl_acc.append(np.corrcoef(x, acc)[0, 1])
        print(f"Correlation between {layer} SMI(L,Y) and accuracy: {np.corrcoef(y, acc)[0, 1]}")
        correlations_ly_acc.append(np.corrcoef(y, acc)[0, 1])

        l_names.append(layer[:-6])
   
        scatter = ax.scatter(x, y, c=acc, cmap=cmap, norm=norm, s=100)
        #scatter eight bit value with star
        eight_bit_row = df[df['bw'] == 8]
        #ax.scatter(eight_bit_row[layer[:-6]+'_mi_xl'].values, eight_bit_row[layer[:-6]+'_mi_ly'].values, c='red', marker='*', s=200, label='8-bit')
        # Set titles and labels
        ax.set_title(f'Layer {layer}')
        ax.set_xlabel('I(X;T)')
        ax.set_ylabel('I(T;Y)')
        ax.grid(True, linestyle='--', alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)
    # Adjust layout
    plt.tight_layout()
    # Save the plot
    #create directory if not exists
    import os
    os.makedirs(f'SMI_score/{dataset}/{model_name}', exist_ok=True)
    plt.savefig(f'SMI_score/{dataset}/{model_name}/inf_plane_plot.png')
    plt.show()

    #create a matrix with correlations, one side is columns, other side is correlation type, correlation values are in cell
    cls = ['SMI(X,L) vs Acc', 'SMI(L,Y) vs Acc']
    corr_matrix = pd.DataFrame(index=l_names, columns=cls)
    corr_matrix['SMI(X,L) vs Acc'] = correlations_xl_acc
    corr_matrix['SMI(L,Y) vs Acc'] = correlations_ly_acc
    corr_matrix.index = l_names
    corr_matrix = corr_matrix.astype(float)
    corr_matrix = corr_matrix.astype(float)
    corr_matrix = corr_matrix.round(4)
    #heatmap matrix
    plt.figure(figsize=(64, 8))
    sns.heatmap(corr_matrix.T, annot=True, fmt=".4f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    plt.xlabel('Layers')
    plt.ylabel('Correlation Types')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'SMI_score/{dataset}/{model_name}/correlation_matrix.png')
    plt.show()

    return correlations_ly_acc, correlations_xl_acc

