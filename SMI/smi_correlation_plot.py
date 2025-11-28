from models.models import create_model
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def corr_mat(df, model_name):
    ref_row = df[df['bw'] == 8]
    if ref_row.empty:
        raise ValueError("Reference row with bw=8 not found.")

    mod = 'ly'
    # 2. Identify all feature columns ending in '_xl'
    if mod == 'ly':
        xl_cols = [col for col in df.columns if col.endswith('_ly')]
    elif mod == 'xl':
        xl_cols = [col for col in df.columns if col.endswith('_xl')]

    # Create a new DataFrame to store the results for correlation analysis
    corr_df = pd.DataFrame()
    corr_df['acc'] = df['acc'] # Add the accuracy column

    # 3. Calculate the absolute difference for each '_xl' column
    for col in xl_cols:
        # Get the reference value from the bw=8 row
        ref_value = ref_row[col].iloc[0]
        
        # Calculate the absolute difference and create a new column
        diff_col_name = f"{col}_abs_diff"
        corr_df[diff_col_name] = abs(df[col] - ref_value)

    # 4. Compute the correlation matrix
    correlation_matrix = corr_df.corr()

    # 5. Plot the correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True,          # Show the correlation values on the map
        cmap='coolwarm',     # Use a diverging colormap
        fmt=".2f",           # Format annotations to two decimal places
        linewidths=.5
    )
    plt.title('Correlation Matrix: Accuracy vs. Absolute Difference from 8-bit XL Values', pad=20)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    # 6. Save the plot
    save_path = f'SMI/correlation_matrix_{model_name}_{mod}.png'
    plt.savefig(save_path)
    print(f"✅ Correlation matrix plot saved to: {save_path}")

    plt.show()
from matplotlib.lines import Line2D
from scipy.stats import pearsonr # Import pearsonr
def plot_smi_corr():
    # This part for loading and initial filtering remains the same
    # model = create_model('resnet50_gluon', 'imagenet_256x256') # Assumed function
    path = 'RUNS/resnet50_gluon/imagenet_256x256/20250630-171200/results.csv'
    path = 'RUNS/resnet50_gluon/imagenet_256x256/20250703-202643/results.csv'

    #path = 'RUNS/resnet18_gluon/imagenet_256x256/20250630-171200/results.csv'

    #path = 'RUNS/mobilenetv2/imagenet_256x256/20250625-020852/results.csv'
    model_name = 'resnet50_gluon'
    df = pd.read_csv(path)
    df = df[df['bw'].isin([2, 3, 4, 5, 6 ,7 , 8])]
    df = df.drop(columns=['Unnamed: 0', 'layer'])
    
    corr_mat(df, model_name)
    exit()
    column = 'global_pool_flatten_post_act_fake_quantizer_mi_ly'
    column = 'layer1_1_act3_post_act_fake_quantizer_mi_xl'  # Example column name, adjust as needed
    df = df[df[column].notna() & df['acc'].notna()][[column, 'acc', 'bw']]
    df['bw'] = df['bw'].astype(int)

    # --- START: New logic to calculate differences ---

    # 1. Isolate the reference row (bw=8)
    # We assume there is exactly one row for bw=8 as per your request.
    reference_row = df[df['bw'] == 8]
    if reference_row.empty:
        raise ValueError("Reference data for bw=8 not found. Cannot calculate differences.")
    
    # Get the SMI and accuracy values from the reference row
    ref_smi = reference_row[column].iloc[0]
    ref_acc = reference_row['acc'].iloc[0]

    # 2. Create new columns for the difference from the reference point
    df['smi_diff'] =abs(df[column] - ref_smi)
    df['acc_degradation'] = ref_acc - df['acc']  # Values will be <= 0

    #drop columns with accuracy degradation values that are less than 1e-12
    #df = df[df['acc_degradation'] >= 0]
    #drop columns with smi_diff values that are less than 1e-12
    #df = df[df['smi_diff'] >= 4e-2]

    #drop 8 bit row
    df = df[df['bw'] != 8]
    #also 7
    #df = df[df['bw'] != 3]
    #also 6,5
    #df = df[df['bw'] != 6]
    df = df[df['bw'] != 7]
    #df = df[df['bw'] != 5]

    #df = df[df['bw'] == 2]
        
    # --- END: New logic ---

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create the figure and axes for the plot
    plt.figure(figsize=(25, 16))
    ax = plt.gca()

    #df sort by bw
    df = df.sort_values(by='bw')

    # Plot the regression line using the NEW difference columns
    sns.regplot(
        x='smi_diff',
        y='acc_degradation',
        data=df,
        color='darkorange',
        scatter=False,
        line_kws={'linewidth': 7},
        ci=95,
        ax=ax,
    )
    sns.scatterplot(
        x='smi_diff',
        y='acc_degradation',
        data=df,
        hue='bw',
        palette='viridis',
        s=500,#150,
        alpha=0.9,
        edgecolor='w',
        linewidth=0.8,
        ax=ax,
    )


    
        # --- Calculate the correlation coefficient ---
    clean_df = df[['smi_diff', 'acc_degradation']].dropna()
    r, _ = pearsonr(clean_df['smi_diff'], clean_df['acc_degradation'])

    # --- Add the line to the legend with the coefficient in the label ---
    handles, labels = ax.get_legend_handles_labels()

    # Create the label string dynamically
    legend_label = f'Regression Line (r={r:.2f})'

    # Create the proxy artist with the new dynamic label
    proxy_line = Line2D([0], [0], color='darkorange', lw=5, label=legend_label)

    # Add the proxy artist's handle to the list
    #empty handles
    handles = []
    handles.append(proxy_line)

    # Recreate the legend with the new handle included
    
    
    # Plot the scatter points using the NEW difference columns
    #plt.xscale('log')
    #plt.yscale('log')


    # --- Customize the plot for clarity ---
    # Add horizontal and vertical lines to mark the (0,0) origin (our reference point)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)

    #make x and y axis log-scale

    plt.xlim(right= 0.1)
    plt.ylim(top = 0.5)




    #
    


    #make title bold
    #plt.title('Acc. Degradation vs. SMI Change', fontsize=72)
    #plt.title(f'SMI(L,Y) in GlobalAvgPool', fontsize=72)
    plt.xlabel('SMI(X,L) Difference from 8-bit', fontsize=74)
    plt.ylabel('Acc. Degradation', fontsize=74)
    
    ax = plt.gca()

    # --- Create the first legend (e.g., for the scatter plot) ---
    # Get the handles and labels from the scatter plot
    scatter_handles, scatter_labels = ax.get_legend_handles_labels()
    # Create the first legend and store it in a variable
    legend1 = ax.legend(handles=scatter_handles, labels=scatter_labels,
                        title='Bit-width', loc='lower right',
                        fontsize=46, title_fontsize=50)

    # --- Manually add the first legend back to the plot ---
    # This is the crucial step to prevent it from being overwritten
    ax.add_artist(legend1)

    # --- Create the second legend (e.g., for the regression line) ---
    # Assuming 'handles' is the list containing your regression line proxy artist
    # from the previous examples.
    # For example:
    # from matplotlib.lines import Line2D
    # handles = [Line2D([0], [0], color='darkorange', lw=3, label='Regression Line')]

    ax.legend(handles=handles, loc='upper right', fontsize=44) # Create the second legend

    #plt.legend(title='Bit-width', loc='lower right', fontsize=44, title_fontsize=46)
    #add another lefent to upper left with handles
    #plt.legend(handles=handles, loc='upper left', fontsize=44, title_fontsize=46)  # Use the modified handles list
    #add legend for regression line
    

    #set tick font size to 18
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    
    #add a zoom to the plot, add it to upper left part
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper left')
    sns.regplot(
        x='smi_diff',
        y='acc_degradation',
        data=df,
        color='darkorange',
        scatter=False,
        ax=ax_inset,
        line_kws={'linewidth': 7},
    )
    sns.scatterplot(
        x='smi_diff',
        y='acc_degradation',
        data=df,
        hue='bw',
        palette='viridis',
        ax=ax_inset,
        legend=False,
        s=400,

    )

    x1, x2 = 0.0, 0.01
    y1, y2 = -0.01, 0.05
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    ax_inset.tick_params(axis='both', which='major', labelsize=0) # Make ticks smaller
    #no x label and y label
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
    
    

    # Save the plot
    plt.savefig(f'SMI/correlation_plot_diff_{column}.png', bbox_inches='tight')
    plt.show()

# Example of how to run the function
# plot_smi_corr()
if __name__ == "__main__":
    plot_smi_corr()