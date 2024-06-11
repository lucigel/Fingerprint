
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from load_datas import load_data, img_size
from dataAgumentaion import CustomDataset, train_transforms, val_transforms
from model import Model

# Initialize the parser
parser = argparse.ArgumentParser(description='Train the model with specific outputs.')

# Add arguments
parser.add_argument('--HAND', action='store_true', help='Train the model for hand prediction')
parser.add_argument('--GENDER', action='store_true', help='Train the model for gender prediction')
parser.add_argument('--ID', action='store_true', help='Train the model for ID prediction')
parser.add_argument('--FINGER', action='store_true', help='Train the model for finger prediction')

# Parse the arguments
args = parser.parse_args()

choose = []
if args.HAND:
    choose.append("HAND")
if args.GENDER:
    choose.append("GENDER")
if args.ID:
    choose.append("ID")
if args.FINGER:
    choose.append("FINGER")

if not choose:
    choose = ["HAND", "GENDER", "ID", "FINGER"]


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
base_path = os.path.normpath(os.path.abspath(os.getcwd()))
Altered_path = base_path + "/code/data/SOCOFing/Altered/Altered-"
Real_path = base_path + "/code/data/SOCOFing/Real"

Easy_IDs, Easy_genders, Easy_hands, Easy_fingers, Easy_images = load_data(Altered_path+'Easy', train=True)
Medium_IDs, Medium_genders, Medium_hands, Medium_fingers, Medium_images = load_data(Altered_path+'Medium', train=True)
Hard_IDs, Hard_genders, Hard_hands, Hard_fingers, Hard_images = load_data(Altered_path+'Hard', train=True)
Real_IDs, Real_genders, Real_hands, Real_fingers, Real_images = load_data(Real_path, train=False)

Altered_IDs = np.concatenate([Easy_IDs, Medium_IDs, Hard_IDs], axis=0)
Altered_genders = np.concatenate([Easy_genders, Medium_genders, Hard_genders], axis=0)
Altered_hands = np.concatenate([Easy_hands, Medium_hands, Hard_hands], axis=0)
Altered_fingers = np.concatenate([Easy_fingers, Medium_fingers, Hard_fingers], axis=0)
Altered_images = np.concatenate([Easy_images, Medium_images, Hard_images], axis=0)

Altered_images = Altered_images.reshape(-1, 1, img_size, img_size) / 255.0  # Normalize images

del Easy_IDs, Easy_genders, Easy_hands, Easy_fingers, Easy_images
del Medium_IDs, Medium_genders, Medium_hands, Medium_fingers, Medium_images
del Hard_IDs, Hard_genders, Hard_hands, Hard_fingers, Hard_images

# Split data
images_train, images_temp, IDs_train, IDs_temp, genders_train, genders_temp, hands_train, hands_temp, fingers_train, fingers_temp = train_test_split(
    Altered_images, Altered_IDs, Altered_genders, Altered_hands, Altered_fingers, test_size=0.2, random_state=42)

images_val, images_test, IDs_val, IDs_test, genders_val, genders_test, hands_val, hands_test, fingers_val, fingers_test = train_test_split(
    images_temp, IDs_temp, genders_temp, hands_temp, fingers_temp, test_size=0.5, random_state=42)

print("Shapes:                  Feature shape    Label shape")
print("----------------------------------------------------")
print("Full data:              ", Altered_images.shape, Altered_IDs.shape, Altered_genders.shape, Altered_hands.shape, Altered_fingers.shape)
print("Train data:             ", images_train.shape, IDs_train.shape, genders_train.shape, hands_train.shape, fingers_train.shape)
print("Validation data:        ", images_val.shape, IDs_val.shape, genders_val.shape, hands_val.shape, fingers_val.shape)
print("Test data:              ", images_test.shape, IDs_test.shape, genders_test.shape, hands_test.shape, fingers_test.shape)

# Create Dataloader
train_data = TensorDataset(torch.tensor(images_train, dtype=torch.float32),
                           torch.tensor(IDs_train, dtype=torch.long),
                           torch.tensor(genders_train, dtype=torch.float32),
                           torch.tensor(hands_train, dtype=torch.float32),
                           torch.tensor(fingers_train, dtype=torch.long))

val_data = TensorDataset(torch.tensor(images_val, dtype=torch.float32),
                         torch.tensor(IDs_val, dtype=torch.long),
                         torch.tensor(genders_val, dtype=torch.float32),
                         torch.tensor(hands_val, dtype=torch.float32),
                         torch.tensor(fingers_val, dtype=torch.long))

train_dataset = CustomDataset(train_data, transform=train_transforms)
val_dataset = CustomDataset(val_data, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Initialize TensorBoard SummaryWriter
log_dir = os.path.join("logs", "fit", "model_1")
writer = SummaryWriter(log_dir)

# Create model
input_shape = (img_size, img_size)
model = Model(input_shape, choose).to(device)

# Loss function and Optimizer
criterion_ID = nn.CrossEntropyLoss()
criterion_gender = nn.BCEWithLogitsLoss()
criterion_hand = nn.BCEWithLogitsLoss()
criterion_finger = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

num_epochs = 50
best_val_loss = float('inf')
early_stop_count = 0
patience = 5  # Number of epochs to wait for improvement before stopping

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, IDs, genders, hands, fingers in train_loader:
        images, IDs, genders, hands, fingers = images.to(device), IDs.to(device), genders.to(device), hands.to(device), fingers.to(device)
        
        optimizer.zero_grad()
        
        output = model(images)
        
        loss_ID = criterion_ID(output["ID"], IDs)
        loss_gender = criterion_gender(output["GENDER"], genders.unsqueeze(1))
        loss_hand = criterion_hand(output["HAND"], hands.unsqueeze(1))
        loss_finger = criterion_finger(output["FINGER"], fingers)
        
        loss = loss_ID + loss_gender + loss_hand + loss_finger
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, IDs, genders, hands, fingers in val_loader:
            images, IDs, genders, hands, fingers = images.to(device), IDs.to(device), genders.to(device), hands.to(device), fingers.to(device)
            
            output = model(images)
            
            loss_ID = criterion_ID(output["ID"], IDs)
            loss_gender = criterion_gender(output["GENDER"], genders.unsqueeze(1))
            loss_hand = criterion_hand(output["HAND"], hands.unsqueeze(1))
            loss_finger = criterion_finger(output["FINGER"], fingers)
            
            loss = loss_ID + loss_gender + loss_hand + loss_finger
            val_loss += loss.item()
            
            _, predicted = torch.max(output["ID"], 1)
            total += IDs.size(0)
            correct += (predicted == IDs).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    
    # Step the scheduler
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_count = 0
        # Save the best model
        torch.save(model.state_dict(), "best_model.pth")
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered")
            break

writer.close()
    

