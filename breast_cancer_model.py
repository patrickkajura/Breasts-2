import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
data_dir = r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer"
output_dir = r"C:\Users\patri\Documents\Health ai\Multi Cancer\Breast Cancer"
os.makedirs(output_dir, exist_ok=True)

# Define transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# Split dataset into train and validation sets
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(0.8 * dataset_size)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# Apply test transform to validation dataset
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            img = transforms.ToPILImage()(x)
            x = self.transform(img)
        return x, y
        
    def __len__(self):
        return len(self.subset)

val_dataset = TransformDataset(val_dataset, transform=transform_test)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Class names
class_names = full_dataset.classes
print("Classes:", class_names)

# Define the model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # Adjust output layer for 2 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, num_epochs=20):
    best_val_acc = 0.0
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc,
                'class_names': class_names
            }, os.path.join(output_dir, "best_breast_cancer_model.pth"))
            print(f"New best model saved with accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
        print()
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_val_loss,
        'accuracy': epoch_val_acc,
        'class_names': class_names,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, os.path.join(output_dir, "final_breast_cancer_model.pth"))
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies

# Train the model
model, train_losses, train_accs, val_losses, val_accs = train_model(model, num_epochs=20)

# Function to predict on a single image
def predict_single_image(image_path, model_path=None):
    if model_path:
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_percent = confidence.item() * 100
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence_percent:.2f}%")
        
        return predicted_class, confidence_percent

# Example usage for prediction (commented out since we don't have a specific test image)
# Uncomment and provide a path to a test image to make a prediction
# predict_single_image("path_to_test_image.jpg", os.path.join(output_dir, "best_breast_cancer_model.pth"))

print("\nModel training completed successfully!")
print(f"Models saved in: {output_dir}")
print("Files saved:")
print("- best_breast_cancer_model.pth (best performing model)")
print("- final_breast_cancer_model.pth (final model with training history)")