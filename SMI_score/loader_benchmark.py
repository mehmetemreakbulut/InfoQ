import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset, Image as HFImage
# sklearn.model_selection.train_test_split is no longer needed
import numpy as np
import os
import matplotlib.pyplot as plt # Optional: for visualization

# --- Enable Debug Logging ---
import datasets as hf_datasets_logging # Use an alias to avoid conflict if 'datasets' is used as a var later
hf_datasets_logging.logging.set_verbosity_debug()
import logging
logging.basicConfig(level=logging.DEBUG) # Enable general python logging for more info
print("DEBUG logging for 'datasets' library enabled.")

# --- Configuration ---
hf_dataset_name_highres = "benjamin-paine/imagenet-1k-256x256"
# num_subset_samples is kept for reference, as the validation split is expected to be this size.
num_expected_samples = 50000
batch_size = 256
num_workers = 8 # Set to 0 for easier debugging in Colab, can be >0 for performance
random_seed = 42 # For reproducibility (less critical now as no random splitting)
final_highres_size = 224 # e.g., standard ImageNet validation size after crop

# --- Script Execution Control ---
script_should_continue = True
ds_highres_validation = None # Renamed for clarity

# --- 1. Transforms ---
preprocess_highres = transforms.Compose([
    transforms.CenterCrop(final_highres_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_transform_to_batch(batch, preprocess_fn, image_column='image', label_column='label'):
    if image_column not in batch:
        logging.error(f"Image column '{image_column}' not found in batch keys: {list(batch.keys())}")
        raise KeyError(f"Image column '{image_column}' not found in batch keys: {list(batch.keys())}")
    try:
        # Ensure images are PIL before applying transforms, convert to RGB
        processed_images = []
        for img in batch[image_column]:
            if not hasattr(img, 'convert'): # Check if it's a PIL image
                logging.warning(f"Encountered non-PIL image in batch: type {type(img)}. Attempting to proceed.")
            processed_images.append(preprocess_fn(img.convert("RGB") if hasattr(img, 'convert') else img))
        batch[image_column] = processed_images
    except Exception as e:
        logging.error(f"Error during image transformation: {e}. First image type: {type(batch[image_column][0]) if batch[image_column] else 'empty'}")
        raise
    if label_column in batch:
         batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
    return batch

# --- 2. Load FULL Validation Dataset from Hugging Face ---
print(f"--- Loading High-Resolution Validation Dataset ({hf_dataset_name_highres}) ---")
logging.info(f"Attempting to load High-Res Dataset: {hf_dataset_name_highres}, split: validation")
try:
    ds_highres_validation = hf_load_dataset(
        hf_dataset_name_highres,
        split='validation', # Directly load the validation split
    )
    ds_highres_train = hf_load_dataset(
        hf_dataset_name_highres,
        split='train', # Directly load the validation split
    )
    print(f"Full High-Res validation dataset loaded successfully.")
    print(f"Full High-Res validation dataset size: {len(ds_highres_validation)}")
    if len(ds_highres_validation) != num_expected_samples:
        print(f"WARNING: Expected {num_expected_samples} samples in validation set, but found {len(ds_highres_validation)}.")
        logging.warning(f"Expected {num_expected_samples} samples in validation set, but found {len(ds_highres_validation)}.")
    print("High-Res Validation Dataset features:", ds_highres_validation.features)
    logging.info("High-Res validation dataset loaded successfully.")
except Exception as e:
    print(f"Error loading High-Res Hugging Face dataset ({hf_dataset_name_highres}): {e}")
    logging.error(f"Detailed error loading High-Res dataset: {e}", exc_info=True)
    script_should_continue = False


image_column = 'image'
label_column = 'label'
logging.info("Dataset loading appears successful. Proceeding with checks and processing.")

# Basic feature check
if ds_highres_validation is None:
    logging.error(f"High-Res validation dataset object is None. This should not happen if script_should_continue is true.")
    script_should_continue = False
elif image_column not in ds_highres_validation.features or not isinstance(ds_highres_validation.features[image_column], HFImage):
        print(f"CRITICAL ERROR: High-Res validation dataset: Expected image column '{image_column}' (type HFImage) not found. Features: {ds_highres_validation.features}")
        logging.error(f"CRITICAL ERROR: High-Res validation dataset: Expected image column '{image_column}' (type HFImage) not found. Features: {ds_highres_validation.features}")
        script_should_continue = False
elif label_column not in ds_highres_validation.features:
        print(f"CRITICAL ERROR: High-Res validation dataset: Label column '{label_column}' not found. Features: {ds_highres_validation.features}")
        logging.error(f"CRITICAL ERROR: High-Res validation dataset: Label column '{label_column}' not found. Features: {ds_highres_validation.features}")
        script_should_continue = False

if not script_should_continue:
    print("Halting script due to critical feature check errors after attempting to load datasets.")
    logging.warning("Halting script due to critical feature check errors.")

# Stratification and subsetting sections are removed as per request.


print("\n--- Applying Transforms to High-Res Validation Dataset ---")
ds_highres_validation = ds_highres_validation.with_transform(None) # Clear any previous transforms if any
try:
    ds_highres_validation.set_transform(lambda batch: apply_transform_to_batch(batch, preprocess_highres, image_column, label_column))
    ds_highres_train.set_transform(lambda batch: apply_transform_to_batch(batch, preprocess_highres, image_column, label_column))
    print("Applied High-Res transform to High-Res validation dataset.")
except Exception as e:
    print(f"Error setting transform on High-Res validation dataset: {e}")
    logging.error(f"Error setting transform on High-Res validation dataset: {e}", exc_info=True)
    script_should_continue = False


print("\n--- Creating DataLoader for High-Res Validation Dataset ---")
loader_highres_validation = DataLoader(
    ds_highres_validation, batch_size=batch_size, shuffle=False, # Shuffle is typically False for validation
    num_workers=num_workers, pin_memory=True
)
loader_highres_train = DataLoader(
    ds_highres_train, batch_size=batch_size, shuffle=True, # Shuffle is typically False for validation
    num_workers=num_workers, pin_memory=True
)
print('High-Res Validation DataLoader is ready.')
print(f"High-Res Validation Loader size (batches): {len(loader_highres_validation)}")

