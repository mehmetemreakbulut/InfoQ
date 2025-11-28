import numpy as np
import pulp as pl
from copy import deepcopy
import pandas as pd
from collections import Counter
# import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import math
from scipy import optimize
import random
# Assume ORM.py exists in the same directory or is installable
import SMI_score.other_algorithms.ompq.ORM as ORM
import torch
import time
import torch.nn as nn
import torchvision
from SMI_score.other_algorithms.ompq.feature_extract import add_feature_extract


# --- Placeholder Functions ---


# --- Helper Functions for ORT Logic (New Implementation) ---

# MODIFIED get_ort_score AS REQUESTED
def get_ort_score(layer_bit_dict, layer_sensitivities, ALLOWED_BIT_WIDTHS=[2, 4, 8], BASE_BIT_WIDTH=8):
    """
    Calculates scores based on the exact "original formulation" ORT logic provided.
    Score = - [ (b / BASE_BIT_WIDTH) * sum(theta[j] for j>=i) / (length - i) ]
    Maximizing this score minimizes the original ORT objective. Uses direct summation.

    Args:
        layer_bit_dict (dict): Dictionary with layer names as keys (provides indices & length).
                               Assumes keys correspond to indices 0..length-1.
        layer_sensitivities (list or array): List/array of theta values,
                                             theta = [theta_0, theta_1, ..., theta_{length-1}].

    Returns:
        dict: scores[layer_index][bit_width]
    """
    scores = {}
    # Assumes layer_bit_dict represents layers 0 to length-1
    length = len(layer_bit_dict)
    layer_indices = list(range(length))

    if not isinstance(layer_sensitivities, (list, np.ndarray)):
         raise TypeError("layer_sensitivities must be a list or numpy array")
    if len(layer_sensitivities) != length:
        raise ValueError(f"Length of layer_sensitivities ({len(layer_sensitivities)}) "
                         f"must match the number of layers ({length})")

    # Ensure sensitivities are floats
    theta = [float(t) for t in layer_sensitivities]

    # Calculate scores using the direct summation logic
    for i in layer_indices: # Outer loop for layers
        scores[i] = {}
        temp = 0.0
        # Inner loop to calculate sum(theta[j] for j >= i)
        for j in range(i, length):
            temp += theta[j]

        denom = length - i

        # Calculate score for each bit width
        for b in ALLOWED_BIT_WIDTHS:
            bit_ratio = b / BASE_BIT_WIDTH
            # Score is negative of the original ORT objective term
            score_val = (bit_ratio * (temp / denom))
            scores[i][b] = float(score_val)

    print("Calculated ORT Scores (direct summation - first few layers):")
    scores_sample_df = pd.DataFrame({k: scores[k] for k in list(scores.keys())[:5]}).T
    scores_sample_df.index.name = 'Layer Index'
    print(scores_sample_df)
    return scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Seeds
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

def get_sensitivities(net,beta,data_loader, model_name='resnet18_gluon', dataset='imagenet_256x256'):

    try:
        i = 0
        for i,batch in enumerate(data_loader):
            if i == 100:
                break
            data = batch
            images = data['image']
            images = images.to(device)
            n = images.size(0) # Get batch size
            break
    except StopIteration:
        print("Error: Data loader is empty.")
        exit()

    if dataset == 'imagenet_256x256':
        add_feature_extract(net, model_name)
    with torch.no_grad():
        # Call the model's feature_extract method
        # Pass only the image data, as the method signature doesn't expect quant_type
        feature = net.feature_extract(images)
    
    # Process extracted features (convert to numpy, flatten)
    for i in range(len(feature)):
        feature[i] = feature[i].view(n, -1)
        feature[i] = feature[i].data.cpu().numpy()
    print(f"Extracted {len(feature)} feature tensors.")

    # --- Orthogonality Calculation ---
    print("Calculating orthogonality matrix...")
    orthogonal_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                orthogonal_matrix[i][j] = ORM.orm(ORM.gram_linear(feature[i]), ORM.gram_linear(feature[j]))
    #print(orthogonal_matrix)

    print("Orthogonality calculation complete.")
    # --- Optimization Setup ---
    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    theta = []
    gamma = []

    for i in range(len(feature)):
        gamma.append(sum_list(orthogonal_matrix[i], i) )
    print(gamma)
    # Check for NaN/Inf in gamma before calculating theta
    if np.isnan(gamma).any() or np.isinf(gamma).any():
        print("Error: NaN or Inf found in gamma values. Cannot proceed with optimization.")
        exit()

      # e^-x calculation for theta
    for i in range(len(feature)):
        theta.append( 1 * math.exp(-1 * beta * gamma[i]) )
    
    theta = np.array(theta)
    #theta = np.negative(theta)

    return theta
