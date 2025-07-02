import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_NAME = 'resnet34'
DATASET_NAME = 'VOC_cropped'
DATA_DIR = "C:\\Users\\simon\\Downloads\\VocPascal\\VocPascal"
IMAGE_FOLDER_NAME = 'JPEGImages'
IMAGE_DIR = os.path.join(DATA_DIR, IMAGE_FOLDER_NAME, IMAGE_FOLDER_NAME)
TRAIN_FILE = os.path.join(DATA_DIR, 'train_voc.txt')
VAL_FILE = os.path.join(DATA_DIR, 'val_voc.txt')

def parse_voc_file(file_path, image_dir):
    annotations = []
    print(f"Parsing annotation file: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: Annotation file not found at {file_path}")
        return annotations
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            image_id = parts[0]
            class_name = parts[1]
            x1, x2, y1, y2 = map(int, parts[2:6])
            
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            
            if xmin >= xmax or ymin >= ymax:
                continue

            bbox = [xmin, ymin, xmax, ymax]
            image_filename = f"{image_id}.jpg"
            full_image_path = os.path.join(image_dir, image_filename)
            
            if os.path.exists(full_image_path):
                annotations.append({
                    'path': full_image_path,
                    'label': class_name,
                    'bbox': bbox
                })

    print(f"Found {len(annotations)} valid objects with existing image files in {file_path}")
    return annotations

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long() 

        if len(self.features) != len(self.labels):
            raise ValueError("Number of features and labels must match.")
        
        print(f"FeatureDataset loaded: {len(self.features)} samples.")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

class Simple_model(nn.Module):
    def __init__(self, input_size, n1_hidden, n2_hidden, n3_hidden, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.input_layer = nn.Linear(input_size, n1_hidden)
        self.act_hidden1 = nn.ReLU() 

        self.hidden_layer = nn.Linear(n1_hidden, n2_hidden)
        self.act_hidden2 = nn.ReLU() 

        self.hidden_layer2 = nn.Linear(n2_hidden, n3_hidden)
        self.act_hidden3 = nn.ReLU()

        self.output_layer = nn.Linear(n3_hidden, n_classes)

    def forward(self, x):
        x = self.act_hidden1(self.input_layer(x))
        x = self.act_hidden2(self.hidden_layer(x))
        x = self.act_hidden3(self.hidden_layer2(x))
        x = self.output_layer(x) 
        return x

if __name__ == '__main__':
    if not os.path.isdir(IMAGE_DIR):
        print(f"FATAL ERROR: Image directory not found at '{IMAGE_DIR}'")
        exit()

    train_annotations = parse_voc_file(TRAIN_FILE, IMAGE_DIR)
    val_annotations = parse_voc_file(VAL_FILE, IMAGE_DIR)
    
    all_annotations = train_annotations + val_annotations
    
    if not all_annotations:
        print("\nFATAL ERROR: No valid image annotations were loaded.")
        exit()
        
    print(f"Total objects to process: {len(all_annotations)}")

    all_class_names = sorted(list(set(ann['label'] for ann in all_annotations)))
    class_to_idx = {name: i for i, name in enumerate(all_class_names)}
    idx_to_class = {i: name for i, name in enumerate(all_class_names)}
    
    print(f"\nFound {len(all_class_names)} classes: {all_class_names}")
    
    labels_for_rf = np.array([class_to_idx[ann['label']] for ann in all_annotations])

    feat_file = os.path.join(DATA_DIR, f'features_{MODEL_NAME}_{DATASET_NAME}.npy')
    
    load_from_cache = False
    if os.path.exists(feat_file):
        print(f"\nFound cached features file: {feat_file}")
        features = np.load(feat_file)
        dim = features.shape[1]
        if features.shape[0] == len(all_annotations):
            print("Cache size matches annotation count. Loading from cache.")
            load_from_cache = True
        else:
            print(f"Cache size mismatch! (Found {features.shape[0]}, expected {len(all_annotations)}). Regenerating features.")

    if not load_from_cache:
        print("\nStarting feature extraction...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = None
        
        if MODEL_NAME == 'resnet34':
            model = models.resnet34(pretrained=True).to(device)
            model.fc = nn.Identity()
            model.eval()
            dim = 512
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif MODEL_NAME == "dinov2": 
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            model.eval()
            dim = 384
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        elif MODEL_NAME == "clip":
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            model = clip_model.encode_image
            dim = 512
        else:
            raise ValueError(f"Model '{MODEL_NAME}' not recognized.")
            
        n_images = len(all_annotations)
        features = np.zeros((n_images, dim), dtype=np.float32)
        
        with torch.no_grad():
            for i, ann in enumerate(tqdm(all_annotations, desc="Extracting Features")):
                try:
                    full_image = Image.open(ann['path']).convert('RGB')
                    cropped_image = full_image.crop(ann['bbox'])
                    image_tensor = preprocess(cropped_image).unsqueeze(0).to(device)
                    feature_vector = model(image_tensor)
                    features[i, :] = feature_vector.cpu().numpy().flatten()
                except Exception as e:
                    print(f"Error processing {ann['path']} with bbox {ann['bbox']}: {e}")
                    features[i, :] = np.zeros(dim, dtype=np.float32)

        print(f"Saving features to {feat_file}")
        np.save(feat_file, features)
        print("Feature extraction complete.")

    print("\nTraining Random Forest classifier...")
    X = features
    y = labels_for_rf
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"\nRandom Forest Overall Accuracy: {rf_accuracy:.4f}")
    
    print("\nRandom Forest Classification Report:")
    report = classification_report(y_test, y_pred_rf, target_names=[idx_to_class[i] for i in sorted(idx_to_class.keys())], zero_division=0)
    print(report)

    print("\nRandom Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    
    epochs = 10
    batch_size = 64
    train_split = 0.8
    learning_rate = 0.01
    momentum = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_test, y_test)

    train_labels = y_train

    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1. / np.maximum(class_counts, 1)

    sample_weights = class_weights[train_labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).double() 
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count() // 2, pin_memory=True)

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)

    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    input_size = dim 
    n_hidden1 = 8
    n_hidden2 = 16
    n_hidden3 = 256 
    n_output = len(all_class_names)

    model = Simple_model(input_size=input_size,
                        n1_hidden=n_hidden1,
                        n2_hidden=n_hidden2,
                        n3_hidden=n_hidden3, 
                        n_classes=n_output)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    val_losses = []
    val_accuracies = [] 

    print("\n--- Starting MLP Training ---")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long) 

            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad(): 
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long) 

                outputs = model(inputs) 

                val_loss = loss_fn(outputs, labels)
                running_val_loss += val_loss.item()

                _, predicted = torch.max(outputs.data, 1) 
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item() 

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = correct_predictions / total_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')

    print("--- MLP Training Finished ---")

    print("\n--- Evaluating MLP on Test Set ---")
    model.eval()
    y_true_mlp = []
    y_pred_mlp = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true_mlp.extend(labels.cpu().numpy())
            y_pred_mlp.extend(predicted.cpu().numpy())

    print("\nMLP Classification Report:")
    target_names_mlp = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print(classification_report(y_true_mlp, y_pred_mlp, target_names=target_names_mlp, zero_division=0))

    print("\nMLP Confusion Matrix:")
    print(confusion_matrix(y_true_mlp, y_pred_mlp))

    mlp_accuracy = accuracy_score(y_true_mlp, y_pred_mlp)
    print(f"\nMLP Overall Accuracy: {mlp_accuracy:.4f}")
    print("\n\n--- Training SVM Classifier ---")
    svm_model = SVC(kernel='rbf', C=10.0, class_weight='balanced', probability=True, random_state=42)
    
    svm_model.fit(X_train, y_train)

    y_pred_svm = svm_model.predict(X_test)
    
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Validation Accuracy: {svm_accuracy:.4f}")
    
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=[idx_to_class[i] for i in sorted(idx_to_class.keys())], zero_division=0))
    
    print("\nSVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))

    print("\n\n--- Model Accuracy Comparison ---")
    
    model_names = ['Random Forest', 'MLP', 'SVM']
    accuracies = [rf_accuracy, mlp_accuracy, svm_accuracy]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Comparison of Model Accuracies')
    plt.ylim(0, 1.05) 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.show()
