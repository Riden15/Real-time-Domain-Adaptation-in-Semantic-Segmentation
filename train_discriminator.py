import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.discriminator import Discriminator

# Define the loss criterion
criterion = nn.BCEWithLogitsLoss()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def train_discriminator(discriminator, data_loader_source, data_loader_target, optimizer, epochs=1, criterion=criterion,
                        device=device):
    """
    Trains the discriminator model.

    :param discriminator: Discriminator neural network model.
    :param data_loader_source: Data loader for the source domain dataset.
    :param data_loader_target: Data loader for the target domain dataset.
    :param optimizer: Optimizer used for training the discriminator.
    :param epochs: Number of epochs for training.
    :param criterion: Loss function used for training.
    :param device: Device to which the model and data are sent ('cuda', 'mps', 'cpu').
    """

    # Move the discriminator model to the specified device (GPU/CPU)
    discriminator.to(device)
    # Set the discriminator model to training mode (this enables dropout, batch-norm layers if present)
    discriminator.train()

    # Training loop for the specified number of epochs
    for epoch in range(epochs):
        total_loss = 0  # Initialize total loss for the epoch
        total_steps = len(data_loader_source) + len(
            data_loader_target)  # Total steps = number of batches in source and target loaders

        # Create a tqdm progress bar for visualizing progress in each epoch
        progress_bar = tqdm(total=total_steps, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        # Iterate over both source and target domain data loaders
        for source_data, target_data in zip(data_loader_source, data_loader_target):
            # Handle data from the source domain
            _, seg_source = source_data  # Unpack source domain data (images, labels)
            seg_source = seg_source.to(device)  # Move source images to the device
            labels_source = torch.ones(seg_source.size(0), 1, seg_source.size(2), seg_source.size(3)).to(
                device)  # Labels are 1 for source domain data

            # Zero the gradients of the optimizer
            optimizer.zero_grad()
            # Forward pass of source images through the discriminator
            outputs_source = discriminator(seg_source)
            # Calculate loss for source domain data
            loss_source = criterion(outputs_source, labels_source)
            # Backward pass to compute gradients
            loss_source.backward()
            # Accumulate the loss
            total_loss += loss_source.item()

            # Handle data from the target domain
            _, seg_target = target_data  # Unpack target domain data (images, labels), labels are ignored
            seg_target = seg_target.to(device)  # Move target images to the device
            labels_target = torch.zeros(seg_target.size(0), 1, seg_target.size(2), seg_target.size(3)).to(
                device)  # Labels are 0 for target domain data
            # Forward pass of target images through the discriminator
            outputs_target = discriminator(seg_target)
            # Calculate loss for target domain data
            loss_target = criterion(outputs_target, labels_target)
            # Backward pass to compute gradients
            loss_target.backward()
            # Accumulate the loss
            total_loss += loss_target.item()

            # Update the weights of the discriminator
            optimizer.step()
            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar at the end of the epoch
        progress_bar.close()
        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {total_loss / total_steps}")


# Dummy data generator
def generate_dummy_data(batch_size, image_size=(3, 512, 256), num_batches=10, classes=19):
    # Generating random tensors as images
    images = torch.randn(num_batches * batch_size, *image_size)
    # Generating random semantic segmentation labels with shape (batch_size, classes, height, width)
    labels = torch.randint(0, classes, (num_batches * batch_size, classes, *image_size[1:]))
    labels = labels.float()
    print(f"Generated images of shape: {images.shape} and labels of shape: {labels.shape}")
    # Creating a dataset and dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Parameters for dummy data
classes = 6
batch_size = 4
image_size = (3, 512, 256)
num_batches = 10

# Create dummy data loaders for source and target domains
data_loader_source = generate_dummy_data(batch_size, image_size, num_batches, classes=classes)
data_loader_target = generate_dummy_data(batch_size, image_size, num_batches, classes=classes)

# Initialize the discriminator
discriminator = Discriminator(in_channels=classes)
optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))

# Run a dummy training iteration
train_discriminator(discriminator, data_loader_source, data_loader_target, optimizer, epochs=5, criterion=criterion,
                    device=device)
