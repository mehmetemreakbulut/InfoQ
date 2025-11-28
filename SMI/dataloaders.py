import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset, Image as HFImage
from sklearn.model_selection import train_test_split
import numpy as np
import os

import numpy as np
#TODO SET ALL SEEDS
image_column = 'image'
label_column = 'label'
random_seed=42

def load_imagenet_256x256_validation(val_split=0.02):
    hf_dataset_name_highres = "benjamin-paine/imagenet-1k-256x256"
    ds_val = hf_load_dataset(hf_dataset_name_highres, split='train', verification_mode='no_checks')
    # Split the dataset into training and validation sets
    ds_train, ds_validation = ds_val.train_test_split(test_size=val_split, seed=random_seed).values()
    transform = transforms.Compose([
        transforms.CenterCrop(224),  # Assuming final size is 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    def apply_transform_to_batch(batch, preprocess_fn, image_column='image', label_column='label'):
        if isinstance(batch[image_column], list):
            batch[image_column] = [preprocess_fn(img.convert("RGB")) for img in batch[image_column]]
        else:
            batch[image_column] = preprocess_fn(batch[image_column].convert("RGB"))
        if label_column in batch:
            batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
        return batch
    ds_train.set_transform(lambda batch: apply_transform_to_batch(batch, transform, image_column, label_column))
    ds_validation.set_transform(lambda batch: apply_transform_to_batch(batch, transform, image_column, label_column))

    val_loader = DataLoader(
        ds_validation,
        batch_size=64,  # Adjust batch size as needed, 32 for Resnet50
        shuffle=True,  # Typically False for validation
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )
    return val_loader

def load_imagenet_256x256_ds(num_target_classes=80, num_samples_per_class=30, batch_size=32, final_highres_size=224, num_workers=4, random_seed=42, subset=True):
    num_subset_samples_target = num_target_classes * num_samples_per_class # Target total samples
    hf_dataset_name_highres = "benjamin-paine/imagenet-1k-256x256"
    ds_train = hf_load_dataset(hf_dataset_name_highres, split='train', verification_mode='no_checks')
    
    if subset:
        labels_highres = ds_train[label_column]

        # --- 4. Select Classes and Generate Subset Indices ---
        np.random.seed(random_seed) # For reproducibility of class selection and sampling

        all_unique_labels_from_train = sorted(list(set(labels_highres)))

        if len(all_unique_labels_from_train) < num_target_classes:
            print(f"CRITICAL ERROR: The TRAIN dataset contains only {len(all_unique_labels_from_train)} unique classes. Cannot select {num_target_classes} distinct classes.")
            exit()

        selected_classes = np.random.choice(all_unique_labels_from_train, size=num_target_classes, replace=False).tolist()
        

        temp_subset_indices = []
        # Build a map of label to indices from the high-res dataset's labels
        label_to_indices_map = {}
        for i, label_val in enumerate(labels_highres): # labels_highres corresponds to indices 0..len(ds_highres)-1
            if label_val not in label_to_indices_map:
                label_to_indices_map[label_val] = []
            label_to_indices_map[label_val].append(i)

        for target_class in selected_classes:
            if target_class in label_to_indices_map:
                class_indices = list(label_to_indices_map[target_class]) # Create a copy to shuffle
                np.random.shuffle(class_indices) # Shuffle to get random samples

                actual_samples_for_class = min(num_samples_per_class, len(class_indices))
                if actual_samples_for_class < num_samples_per_class:
                    pass #print(f"Warning: Class {target_class} has {len(class_indices)} samples in TRAIN split, which is less than the requested {num_samples_per_class}. Using all {actual_samples_for_class} samples.")

                temp_subset_indices.extend(class_indices[:actual_samples_for_class])
            else:
                # This case should ideally not be hit if selected_classes are derived from labels_highres
                pass #print(f"Warning: Selected class {target_class} not found in label_to_indices_map. This is unexpected.")

        subset_indices = sorted(temp_subset_indices) # Keep indices sorted, as in the original script

        ds_train = ds_train.select(subset_indices)

    transform = transforms.Compose([
        transforms.CenterCrop(final_highres_size),
        transforms.ToTensor(),
    ])
    def apply_transform_to_batch(batch, preprocess_fn, image_column='image', label_column='label'):
        if isinstance(batch[image_column], list): # If it's a list of images (e.g. from batched map)
            batch[image_column] = [preprocess_fn(img.convert("RGB")) for img in batch[image_column]]
        else: # If it's a single image (e.g. from set_transform on individual items)
            batch[image_column] = preprocess_fn(batch[image_column].convert("RGB"))
        if label_column in batch:
            batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
        return batch
    
    ds_train.set_transform(lambda batch: apply_transform_to_batch(batch, transform, image_column, label_column))
    
    return ds_train

def load_imagenet_256x256(num_target_classes = 80, num_samples_per_class = 30, batch_size=32, final_highres_size = 224, num_workers=4, random_seed=42, subset=True):
    num_subset_samples_target = num_target_classes * num_samples_per_class # Target total samples

    
    hf_dataset_name_highres = "benjamin-paine/imagenet-1k-256x256"

    transform = transforms.Compose([
    # Input is already 256x256 PIL Image
    transforms.CenterCrop(final_highres_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


    ds_train = hf_load_dataset(hf_dataset_name_highres, split='train', verification_mode='no_checks')
    ds_validation = hf_load_dataset(hf_dataset_name_highres, split='validation')

    image_column = 'image'
    label_column = 'label'

    if subset:
        labels_highres = ds_train[label_column]

        # --- 4. Select Classes and Generate Subset Indices ---
        np.random.seed(random_seed) # For reproducibility of class selection and sampling

        all_unique_labels_from_train = sorted(list(set(labels_highres)))

        if len(all_unique_labels_from_train) < num_target_classes:
            print(f"CRITICAL ERROR: The TRAIN dataset contains only {len(all_unique_labels_from_train)} unique classes. Cannot select {num_target_classes} distinct classes.")
            exit()

        selected_classes = np.random.choice(all_unique_labels_from_train, size=num_target_classes, replace=False).tolist()

        temp_subset_indices = []
        # Build a map of label to indices from the high-res dataset's labels
        label_to_indices_map = {}
        for i, label_val in enumerate(labels_highres): # labels_highres corresponds to indices 0..len(ds_highres)-1
            if label_val not in label_to_indices_map:
                label_to_indices_map[label_val] = []
            label_to_indices_map[label_val].append(i)

        for target_class in selected_classes:
            if target_class in label_to_indices_map:
                class_indices = list(label_to_indices_map[target_class]) # Create a copy to shuffle
                np.random.shuffle(class_indices) # Shuffle to get random samples

                actual_samples_for_class = min(num_samples_per_class, len(class_indices))
                if actual_samples_for_class < num_samples_per_class:
                    pass #print(f"Warning: Class {target_class} has {len(class_indices)} samples in TRAIN split, which is less than the requested {num_samples_per_class}. Using all {actual_samples_for_class} samples.")

                temp_subset_indices.extend(class_indices[:actual_samples_for_class])
            else:
                # This case should ideally not be hit if selected_classes are derived from labels_highres
                pass #print(f"Warning: Selected class {target_class} not found in label_to_indices_map. This is unexpected.")

        subset_indices = sorted(temp_subset_indices) # Keep indices sorted, as in the original script

        ds_train = ds_train.select(subset_indices)

    image_column = 'image'
    label_column = 'label'
    
    def apply_transform_to_batch(batch, preprocess_fn, image_column='image', label_column='label'):
        if isinstance(batch[image_column], list): # If it's a list of images (e.g. from batched map)
            batch[image_column] = [preprocess_fn(img.convert("RGB")) for img in batch[image_column]]
        else: # If it's a single image (e.g. from set_transform on individual items)
            batch[image_column] = preprocess_fn(batch[image_column].convert("RGB"))
        if label_column in batch:
            batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
        return batch

    ds_train = ds_train.with_transform(None)
    ds_validation = ds_validation.with_transform(None)
    ds_train.set_transform(lambda batch: apply_transform_to_batch(batch, transform, image_column, label_column))
    print("Applied High-Res transform to High-Res subset.")

    ds_validation.set_transform(lambda batch: apply_transform_to_batch(batch, transform, image_column, label_column))

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False, # Typically False for validation/testing, can be True for training subset if needed
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        ds_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    train_loader_shuffled = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True, # Typically False for validation/testing, can be True for training subset if needed
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, validation_loader, train_loader_shuffled


def load_cifar10(subset_size=None, batch_size=32):
    hf_dataset_name = 'uoft-cs/cifar10'
    ds = hf_load_dataset(hf_dataset_name, split='train', verification_mode='no_checks')
    ds_test = hf_load_dataset(hf_dataset_name, split='test', verification_mode='no_checks')
    ds = ds.rename_column('img', 'image')
    ds_test = ds_test.rename_column('img', 'image')
    image_column = 'image'
    label_column = 'label'

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    def preprocess_batch(batch):
        if isinstance(batch[image_column], list):
            batch[image_column] = [preprocess(img.convert("RGB")) for img in batch[image_column]]
        else:
            batch[image_column] = preprocess(batch[image_column].convert("RGB"))
        if label_column in batch:
            batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
        return batch

    
        

    ds.set_transform(lambda batch: preprocess_batch(batch))
    ds_test.set_transform(lambda batch: preprocess_batch(batch))

    if subset_size is not None:
        #get subset train_test_split
        ds_subset = ds.train_test_split(train_size=subset_size, seed=42)['train']
        ds = ds_subset

    #change image column from img to image

    train_dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    test_dataloader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    train_dataloader_shuffled = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, test_dataloader, train_dataloader_shuffled


def get_dataloader(dataset, num_target_classes = 80, num_samples_per_class = 30, cifar_subset_size=None, batch_size=32,  num_workers=4, subset=True):
    if dataset == 'imagenet_256x256':
        return load_imagenet_256x256(num_target_classes=num_target_classes, num_samples_per_class=num_samples_per_class, batch_size=batch_size, subset=subset, num_workers=num_workers)
    if dataset =='cifar10':
        return load_cifar10(subset_size=cifar_subset_size, batch_size=batch_size)
    else:
        raise ValueError(f"Dataset {dataset} is not supported. Supported datasets are: 'imagenet_256x256', 'cifar10'.")



def get_targets_imagenet(loader, num_target_classes, num_samples_per_class ):
    import numpy as np
    #set random seed
    np.random.seed(42)  # For reproducibility
    num_subset_samples = num_target_classes * num_samples_per_class  # Total samples expected in the subset
    label_column = 'label'  # Default label column name, adjust if necessary
    import torch

    all_labels_tensors = []
    # Ensure deterministic order if shuffle was ever True (it's False in the setup)
    # subset_loader_highres.shuffle = False # Not directly settable like this after init

    with torch.no_grad(): # Important: disables gradient calculation
        for i, batch in enumerate(loader):
            if i % 50 == 0: # Print progress update
                print(f"  Processing batch {i}/{len(loader)}...")
            try:
                # Ensure labels are on CPU for numpy conversion
                labels = batch[label_column].cpu()
                all_labels_tensors.append(labels)
            except KeyError:
                print(f"Error: Label column '{label_column}' not found in batch keys: {batch.keys()}")
                # Decide how to handle: break, continue, raise error
                break
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                break # Stop processing on error


    if all_labels_tensors: # Proceed only if labels were collected
        # Concatenate all batch tensors into one single tensor
        full_labels_tensor = torch.cat(all_labels_tensors, dim=0)

        # Convert the final tensor to a NumPy array
        targets_np = full_labels_tensor.numpy() # Shape will be (num_subset_samples,)


        # Reshape the array to (num_subset_samples, 1)
        targets_np_reshaped = targets_np.reshape(-1, 1)

        # Verification
        if targets_np_reshaped.shape == (num_subset_samples, 1):
            print("Shape is correct (num_subset_samples, 1).")
        else:
            print(f"Warning: Shape {targets_np_reshaped.shape} doesn't match expected ({num_subset_samples}, 1).")
    else:
        print("No labels were collected from the DataLoader.")

    #change labels lets say, from 136 to 0, 411 to 1, 513 to 2, 521 to 4, and so on for 626, 660, 678, 737, 740, 850 TO 10
    unique_labels = np.unique(targets_np_reshaped)
    print(unique_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    new_targets = np.zeros_like(targets_np_reshaped)
    for i in range(len(targets_np_reshaped)):
        new_targets[i] = label_mapping[targets_np_reshaped[i][0]]
    
    return new_targets

def get_targets(loader, dataset, num_target_classes=80, num_samples_per_class=30):
    if dataset=='cifar10':
        #reshape the targets to (num_subset_samples, 1)
        reshaped_targets = []
        for batch in loader:
            labels = batch['label'].numpy()  # Assuming 'label' is the key for labels
            reshaped_targets.append(labels.reshape(-1, 1))  # Reshape each batch's labels
        reshaped_targets = np.concatenate(reshaped_targets, axis=0)  # Concatenate all batches
        return reshaped_targets
    elif dataset == 'imagenet_256x256':
        return get_targets_imagenet(loader, num_target_classes, num_samples_per_class)
    