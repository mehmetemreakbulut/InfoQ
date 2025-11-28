import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
  
def evaluate_model(model, dataloader):
  #evaluate resnet18 using dataloader val_loader
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)
  # Set the model to evaluation mode
  model.eval()
  #use resnet18
  true_labels = []
  predicted_labels = []
  i  = 0
  image_column = 'image'
  label_column = 'label'
  for batch in tqdm(dataloader, desc="Running Evaluation"):
    images = batch[image_column].to(device)
    labels = batch[label_column].to(device)
    with torch.no_grad():
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      true_labels.extend(labels.cpu().numpy())
      predicted_labels.extend(predicted.cpu().numpy())

  #calculate accuracy
  accuracy = accuracy_score(true_labels, predicted_labels)
  print(f"Accuracy: {accuracy * 100:.2f}%")
  return accuracy
