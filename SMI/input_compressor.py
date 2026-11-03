from PIL import Image
import numpy as np
import requests
from transformers import AutoImageProcessor, AutoModel
import torch
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"

image_column = 'image'
label_column = 'label'

def dinov2_model():
    model_id = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device) # Move the model to the device
    model.eval() # Set to evaluation mode

    # --- Load an image ---
    # (Replace with your ImageNet image loading)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        raise e
    
    # --- Process the image ---
    inputs = processor(images=image, return_tensors="pt").to(device)

    # --- Get model outputs ---
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)

    # --- Extract the features ---
    # For DINOv2, the representation is often taken from the CLS token's
    # output in the last hidden state.
    # The CLS token is the first token in the sequence.
    last_hidden_state = outputs.last_hidden_state
    image_features_384 = last_hidden_state[:, 0] # Shape: [1, 768]

    print("DINOv2 Base Output Shape:", image_features_384.shape)

    return processor, model

def check_dinov2_inputs(processor, ds, processor_kwargs, num_images=64):
    """Fails if the processed DINOv2 inputs are not ImageNet-normalized (e.g. rescaled by 1/255 twice)."""
    images = [ds[i][image_column] for i in range(min(num_images, len(ds)))]
    pixel_values = processor(images=images, return_tensors="pt", **processor_kwargs)['pixel_values']
    pmin, pmax, pstd = pixel_values.min().item(), pixel_values.max().item(), pixel_values.std().item()
    print(f"DINOv2 input check on {len(images)} images: min={pmin:.3f} max={pmax:.3f} std={pstd:.3f}")
    assert pstd > 0.5 and pmax > 1.0, "DINOv2 inputs are not normalized once with ImageNet mean/std"


def imagenet_embeddings(processor, model, ds, fix_preprocessing=False):
    image_column = 'image'
    label_column = 'label'
    X_compressed = []
    model.to(device)
    # ds already yields 224x224 center crops scaled to [0, 1] (ToTensor): only normalize here.
    # fix_preprocessing=False keeps the original behaviour (processor rescales by 1/255 again, resizes and crops).
    processor_kwargs = dict(do_rescale=False, do_resize=False, do_center_crop=False) if fix_preprocessing else {}
    if fix_preprocessing:
        check_dinov2_inputs(processor, ds, processor_kwargs)
    with torch.no_grad():
      i = 0
      for i, batch in tqdm(enumerate(ds), desc="Running Inference"):
          # Get images and potentially labels from the batch
          # Use the correct key for your image data (e.g., 'image')
          image = batch[image_column]


          inputs = processor(images=image, return_tensors="pt", **processor_kwargs).to(device)
          #latent_sample = model.get_image_features(**inputs) # Shape: [1, 512]
          latent_sample = model(**inputs) # Shape: [1, 512]

          last_hidden_state = latent_sample.last_hidden_state
          latent_sample = last_hidden_state[:, 0] # Shape: [1, 768]

          X_compressed.extend(latent_sample.cpu().numpy())
          #scaling_factor = vae.config.scaling_factor
          #latents_for_decoder = latent_sample / scaling_factor
          #decoded_output = vae_dc.decode(latent_dist)
    spread = np.asarray(X_compressed).std(axis=0).mean()
    print(f"X embeddings: {len(X_compressed)} samples, mean per-dimension std across samples = {spread:.4f}")
    return X_compressed


#----------------------#-----------------------#

from .compress_models import Autoencoder, CIFAR10_ConvEncoder, CIFAR10_ConvDecoder, get_outputs
#from . import smi_calculation



def input_embeddings_cifar10(loader):
    X_latent_dim = 100
    X_autoencoder = Autoencoder(CIFAR10_ConvEncoder(latent_dim=X_latent_dim), CIFAR10_ConvDecoder(latent_dim=X_latent_dim)).to(device)
    
    X_autoencoder = torch.load('SMI/all_autoencoder_4', weights_only=False)
    X_autoencoder.agn.enabled_on_inference = False
    X_compressed = get_outputs(X_autoencoder.encoder, loader, device).numpy()
    return X_compressed
    X_compressed = []
    print(loader.dataset)
    #only flatten cifar10 images
    for batch in tqdm(loader, desc="Running Inference"):
        images = batch['image']  # Assuming 'image' is the key for images in the batch
        images = images.view(images.size(0), -1)  # Flatten each image
        X_compressed.extend(images.cpu().numpy())
    return X_compressed

from SMI.vae import load_model_with_fc
from torchvision import transforms
from torch.utils.data import DataLoader
def imagenet_256x256_X(ds, fix_preprocessing=False):
    '''
    print("Loading VAE model with FC layer...")
    vae = load_model_with_fc()
    tr = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def apply_transform_to_batch(batch, preprocess_fn, image_column='image', label_column='label'):
        if isinstance(batch[image_column], list): # If it's a list of images (e.g. from batched map)
            batch[image_column] = [preprocess_fn(img.convert("RGB")) for img in batch[image_column]]
        else: # If it's a single image (e.g. from set_transform on individual items)
            batch[image_column] = preprocess_fn(batch[image_column].convert("RGB"))
        if label_column in batch:
            batch[label_column] = torch.tensor(batch[label_column], dtype=torch.long)
        return batch
    ds.set_transform(lambda batch: apply_transform_to_batch(batch, tr))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    X_compressed = []
    for batch in tqdm(loader, desc="Running Inference"):
        images = batch['image']  # Assuming 'image' is the key for images in the batch
        images = images.to(device)  # Move to device
        with torch.no_grad():
            latent_sample = vae.encoder(images)
            latent_sample = vae.fc(latent_sample)
            latent_sample = latent_sample.view(latent_sample.size(0), -1)  #
        X_compressed.extend(latent_sample.cpu().numpy())
    return X_compressed
    '''
    processor, model = dinov2_model()
    return imagenet_embeddings(processor, model, ds, fix_preprocessing=fix_preprocessing)
    
def get_X(dataset, loader):
    if dataset == 'imagenet_256x256':
        processor, model = dinov2_model()
        return imagenet_embeddings(processor, model, loader)
    elif dataset == 'cifar10':
        return input_embeddings_cifar10(loader)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")