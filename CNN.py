import cv2
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomRotation,
    RandomAffine,
    RandomHorizontalFlip,
    RandomCrop,
    RandomApply,
    ColorJitter,
    GaussianBlur,
)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from digits import show_digits_boundry

# ------------------------------ preproccessing -----------------------------
# Define the training transformations
transform_train = Compose(
    [
        # RandomCrop(29, padding=4),
        # RandomApply([GaussianBlur(3, sigma=(0.1, 2))], p=0.5),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        RandomRotation(degrees=(-20, 20)),
        # RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.13072014), (0.30810788)),
    ]
)

# For validation set: no rotation
transform = Compose([ToTensor(), Normalize((0.13072014), (0.30810788))])

# train data
train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_train,
)

# test data
test_data = MNIST(root="data", train=False, download=True, transform=transform)
# print(train_data[0][0].shape)

# Define constants
train_batch_size = 64
test_batch_size = 64
epochs = 5
lr = 0.001
# momentum is designed to accelerate learning for SGD
momentum = 0.05


train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []
test_accuracies = []
test_losses = []
# Define the paths for saving the model and optimizer state.
modelPath = "pth/m.pth"   # Path where the trained model will be saved.
optimizerPath = "pth/o.pth"  # Path where the optimizer state will be saved.

# Initialize a random generator with a fixed seed for reproducibility.
generator1 = torch.Generator().manual_seed(42)

# Split the training data into training and validation datasets using an 80-20 split.
traindata, valdata = random_split(train_data, [0.8, 0.2], generator=generator1)

# Create a DataLoader for the training data.
train_loader = DataLoader(traindata, batch_size=train_batch_size, shuffle=True)

#set costum transform
valdata.dataset.transform = transform
# Validate DataLoader
val_loader = DataLoader(valdata, batch_size=train_batch_size, shuffle=False)

# Test DataLoader
test_loader = DataLoader(test_data, batch_size=test_batch_size)


# Define Neural Network class
class NN_CNN(nn.Module):
    def __init__(self):
        super(NN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(320, 220)
        self.fc2 = nn.Linear(220, 150)
        self.fc3 = nn.Linear(150, 10)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv2(F.max_pool2d(F.relu(self.conv1(x)), 2))), 2)
        x = x.view(-1, 320)
        x = self.fc3(self.drop(F.relu(self.fc2(self.drop(F.relu(self.fc1(x)))))))
        x = F.relu(x)
        return x

#create an instance of our neural CNN model
model = NN_CNN()
# print(model)

#Define optimazer and loss function
loss = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=lr)
#
def train_model(model, trainloader, loss, optimizer, epochs):
    # Initialize training losses and accuracies.
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        # Variables  of correct predictions and total predictions.
        correct = 0
        total = 0

        # Iterate over the training data.
        for image, labels in trainloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            #compute the model output.
            outputs = model(image)
            # Calculate the loss
            train_loss = loss(outputs, labels)
            #compute gradient of the loss with respect to model parameters.
            train_loss.backward()
            #optimization step to update model parameters.
            optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and store accuracy for this epoch.
        accuracy = correct / total
        train_accuracies.append(accuracy)
        # Save model and optimizer states.
        torch.save(model.state_dict(), modelPath)
        torch.save(optimizer.state_dict(), optimizerPath)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train_Accuracy: {accuracy * 100:.2f}% , Train_Loss: {train_loss}")

        # validation section
        val_accuracies, val_losses = evaluate_model(model, val_loader)
        train_losses.append(train_loss)
    return train_losses, train_accuracies, val_accuracies, val_losses
def evaluate_model(model, dataloader, val=True):
    correct = 0
    total = 0
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the data in the given dataloader.
        for inputs, labels in dataloader:
            # Forward pass: compute the model output.
            outputs = model(inputs)
            # Compute loss for logging purposes(not used for model update)
            losses = loss(outputs, labels)
            # Get predictions from the maximum value of the output
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(total)
    # print(correct)
    # Calculate accuracy
    accuracy = correct / total
    # Conditionally log for validation or test
    if val:
        val_accuracies.append(accuracy)
        val_losses.append(losses)
        print(f"Validation_Accuracy: {accuracy * 100:.2f}% , Validation_Loss: {losses}")
        return val_accuracies, val_losses
    else:
        # test_accuracies.append(accuracy)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        # return test_accuracies
# plot_metrics
def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses.detach().numpy(), label="Training Loss")
    plt.plot(val_losses.detach().numpy(), label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
    plt.title("Training and Validation Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
def showPlot(train_losses, train_accuracies, val_losses, val_accuracies):
    train_losses_tensor = torch.tensor(train_losses)
    train_accuracies_np = np.array(train_accuracies)
    val_losses_tensor = torch.tensor(val_losses).clone().detach().requires_grad_(True)

    val_accuracies_np = np.array(val_accuracies)
    plot_metrics(
        train_losses_tensor, train_accuracies_np, val_losses_tensor, val_accuracies_np
    )
def train():
    # for train data and validation
    train_losses, train_accuracies, val_accuracies, val_losses = train_model(
        model, train_loader, loss, optimizer, epochs
    )
    # for test data
    evaluate_model(model, test_loader, val=False)
    showPlot(train_losses, train_accuracies, val_losses, val_accuracies)
def load_trained_net(images, originalImage, rectangles):
    print("load train!")
    network_state_dict = torch.load(modelPath)
    model.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load(optimizerPath)
    optimizer.load_state_dict(optimizer_state_dict)
    predict_digits(images, originalImage, rectangles)
def predictSingleImage(image):
    model.eval()
    # The requires_grad argument tells PyTorch that we want to be able to calculate the gradients for those values.
    # However, the with torch.no_grad() tells PyTorch to not calculate the gradients

    # Normalize the image using the same parameters as your training data
    normalized_img = (image - 0.13072014) / 0.30810788

    with torch.no_grad():
        single_loaded_img = normalized_img  # test_loader.dataset.data[0]

        single_loaded_img = single_loaded_img[None, None]
        single_loaded_img = torch.from_numpy(single_loaded_img)
        single_loaded_img = single_loaded_img.type(
            "torch.FloatTensor"
        )  # instead of DoubleTensor

        out_predict = model(single_loaded_img)
        pred = out_predict.max(1, keepdim=True)[1]
        return pred.item()
def predict_digits(images, originalImage, rectangles):
    predict = []
    for i in range(len(images)):
        img = images[i]
        img = cv2.dilate(img, kernel=(5, 5), iterations=1)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.addWeighted(img, 2, 0, 0, 0)
        img = cv2.resize(img, (28, 28))
        img = cv2.blur(img, (2, 2), 0)
        img = cv2.addWeighted(img, 2, 0, 0, 0)
        # print("after resize !!!")
        # im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(im2)
        # plt.show()
        predict.append(predictSingleImage(img))
    print("predicted  ====> ", predict)
    show_digits_boundry(originalImage, rectangles, predicted=predict)
    return predict
