import torch
import torch.nn as nn
import numpy as np
import copy
from ..models.resnet import resnet18, BasicBlock


def split_retrain(x_train, x_test, y_train, y_test, target_class, num_cls=10, num_neg_samples=None):
    """
    Resample dataset for specific classes for pruning.

    Args:
        x_train, y_train: Training data and labels
        x_test, y_test: Test data and labels
        target_class: List of target classes to focus on
        num_cls: Total number of classes (default: 10 for CIFAR-10)
        num_neg_samples: Number of negative samples per non-target class (default: auto calculate)

    Returns:
        x_retrain_train, y_retrain_train, x_retrain_test, y_retrain_test
    """
    # Auto calculate number of negative samples per non-target class
    # Each class has len(y_train)/num_cls samples, minus the number of target classes
    if num_neg_samples is None:
        num_neg_samples = int(len(y_train) / num_cls - len(target_class))

    # CIFAR-10 normalization parameters
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    # Convert to float32 and normalize to [0, 1] if needed
    if x_train.dtype == torch.uint8:
        x_train = x_train.float() / 255.0
        x_test = x_test.float() / 255.0

    # Convert NHWC to NCHW if necessary (CIFAR-10 data is NHWC format)
    if len(x_train.shape) == 4 and x_train.shape[-1] == 3:
        # NHWC -> NCHW
        x_train = x_train.permute(0, 3, 1, 2)
        x_test = x_test.permute(0, 3, 1, 2)

    # Apply CIFAR-10 normalization
    # Reshape mean and std for broadcasting [C] -> [C, 1, 1]
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Create label mapping for target classes (1, 2, 3, ...)
    mapping = {k: v for k, v in zip(target_class, range(1, len(target_class) + 1))}

    # Number of negative samples to keep per non-target class
    length = num_neg_samples

    # Initialize empty tensors for retrain dataset
    x_retrain_train = torch.zeros(0, *x_train.shape[1:], dtype=x_train.dtype)
    y_retrain_train = torch.zeros(0, dtype=torch.long)
    x_retrain_test = torch.zeros(0, *x_test.shape[1:], dtype=x_test.dtype)
    y_retrain_test = torch.zeros(0, dtype=torch.long)

    for i in range(num_cls):
        if i in target_class:
            # Positive samples: target class
            new_label = mapping[i]

            # Training set
            positive_image = x_train[y_train == i]
            x_retrain_train = torch.cat([x_retrain_train, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_train[y_train == i]), dtype=torch.long)
            y_retrain_train = torch.cat([y_retrain_train, positive_label], dim=0)

            # Test set
            positive_image = x_test[y_test == i]
            x_retrain_test = torch.cat([x_retrain_test, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_test[y_test == i]), dtype=torch.long)
            y_retrain_test = torch.cat([y_retrain_test, positive_label], dim=0)

        else:
            # Negative samples: non-target class (label = 0)
            # Limit the number of negative samples to 'length' per class

            # Training set
            negative_image = x_train[y_train == i]
            neg_count = min(len(negative_image), length)
            rnd_idx = torch.randperm(len(negative_image))[:neg_count]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros(neg_count, dtype=torch.long)
            x_retrain_train = torch.cat([x_retrain_train, negative_image], dim=0)
            y_retrain_train = torch.cat([y_retrain_train, negative_label], dim=0)

            # Test set
            negative_image = x_test[y_test == i]
            neg_count = min(len(negative_image), length)
            rnd_idx = torch.randperm(len(negative_image))[:neg_count]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros(neg_count, dtype=torch.long)
            x_retrain_test = torch.cat([x_retrain_test, negative_image], dim=0)
            y_retrain_test = torch.cat([y_retrain_test, negative_label], dim=0)

    return x_retrain_train, y_retrain_train, x_retrain_test, y_retrain_test


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for pruned training/testing.
    Labels: 0 = negative (non-target), 1+ = positive (target classes)
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class APoZ:
    """
    APoZ (Average Percentage of Zeros) - A channel pruning method that removes channels
    with high zero activation percentage.
    """
    def __init__(self, model):
        self.model = model
        self.activation = {}
        self.conv_layers = []
        self.hooks = []
        self.register_hooks()

    def get_activation(self, name):
        def hook(module, input, output):
            self.activation[name] = output.detach()
        return hook

    def register_hooks(self):
        # Register forward hooks to gather activation outputs
        # First layer (outside of BasicBlocks)
        self.conv_layers.append('conv1')
        self.hooks.append(self.model.relu.register_forward_hook(
            self.get_activation('conv1')))

        # Layer 1 (2 BasicBlocks)
        for i in range(2):
            self.conv_layers.append(f'layer1.{i}.conv1')
            self.hooks.append(self.model.layer1[i].relu.register_forward_hook(
                self.get_activation(f'layer1.{i}.conv1')))

            self.conv_layers.append(f'layer1.{i}.conv2')
            self.hooks.append(self.model.layer1[i].bn2.register_forward_hook(
                self.get_activation(f'layer1.{i}.conv2')))

        # Layer 2 (2 BasicBlocks)
        for i in range(2):
            self.conv_layers.append(f'layer2.{i}.conv1')
            self.hooks.append(self.model.layer2[i].relu.register_forward_hook(
                self.get_activation(f'layer2.{i}.conv1')))

            self.conv_layers.append(f'layer2.{i}.conv2')
            self.hooks.append(self.model.layer2[i].bn2.register_forward_hook(
                self.get_activation(f'layer2.{i}.conv2')))

        # Layer 3 (2 BasicBlocks)
        for i in range(2):
            self.conv_layers.append(f'layer3.{i}.conv1')
            self.hooks.append(self.model.layer3[i].relu.register_forward_hook(
                self.get_activation(f'layer3.{i}.conv1')))

            self.conv_layers.append(f'layer3.{i}.conv2')
            self.hooks.append(self.model.layer3[i].bn2.register_forward_hook(
                self.get_activation(f'layer3.{i}.conv2')))

        # Layer 4 (2 BasicBlocks)
        for i in range(2):
            self.conv_layers.append(f'layer4.{i}.conv1')
            self.hooks.append(self.model.layer4[i].relu.register_forward_hook(
                self.get_activation(f'layer4.{i}.conv1')))

            self.conv_layers.append(f'layer4.{i}.conv2')
            self.hooks.append(self.model.layer4[i].bn2.register_forward_hook(
                self.get_activation(f'layer4.{i}.conv2')))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_apoz(self, data_loader, num_batches=100):
        """
        Compute APoZ for each channel in the model using a subset of data
        """
        # Initialize dictionary to store APoZ values
        apoz_values = {}
        batch_count = 0

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for inputs, _ in data_loader:
                if batch_count >= num_batches:
                    break

                inputs = inputs.to(device)
                self.model(inputs)

                # Calculate APoZ for each layer
                for layer_name in self.conv_layers:
                    if layer_name not in apoz_values:
                        apoz_values[layer_name] = []

                    act = self.activation[layer_name]

                    # Calculate percentage of zeros in each channel
                    # Shape: [N, C, H, W] -> [C]
                    zero_percentage = (act == 0).sum(dim=(0, 2, 3)).float() / (act.shape[0] * act.shape[2] * act.shape[3])

                    if len(apoz_values[layer_name]) == 0:
                        apoz_values[layer_name] = zero_percentage.cpu().numpy()
                    else:
                        apoz_values[layer_name] += zero_percentage.cpu().numpy()

                batch_count += 1

        # Average the APoZ values over all batches
        for layer_name in apoz_values:
            apoz_values[layer_name] /= batch_count

        return apoz_values

    def compute_class_specific_apoz(self, data_loader, target_classes=None, num_batches=100):
        """
        Compute class-specific APoZ for each channel in the model using a subset of data
        """
        # Initialize dictionaries to store APoZ values
        target_apoz_values = {}
        target_batch_count = 0

        # Convert target_classes to a set for faster lookup
        if target_classes is None:
            raise ValueError("Target classes must be specified for class-specific pruning")

        target_classes_set = set(target_classes)

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for inputs, labels in data_loader:
                if target_batch_count >= num_batches:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Split batch by target classes
                target_indices = torch.tensor([i for i, label in enumerate(labels)
                                            if label.item() in target_classes_set])

                # Skip empty batches
                if len(target_indices) == 0:
                    continue

                # Forward pass
                self.model(inputs)

                # Process target samples
                if len(target_indices) > 0:
                    for layer_name in self.conv_layers:
                        if layer_name not in target_apoz_values:
                            target_apoz_values[layer_name] = []

                        act = self.activation[layer_name]
                        # Select only activations from target class samples
                        target_act = act[target_indices]

                        if len(target_act) > 0:  # Ensure we have samples
                            # Calculate percentage of zeros in each channel for target class
                            zero_percentage = (target_act == 0).sum(dim=(0, 2, 3)).float() / (target_act.shape[0] * target_act.shape[2] * target_act.shape[3])

                            if len(target_apoz_values[layer_name]) == 0:
                                target_apoz_values[layer_name] = zero_percentage.cpu().numpy()
                            else:
                                target_apoz_values[layer_name] += zero_percentage.cpu().numpy()

                    target_batch_count += 1

        # Average the APoZ values over all processed batches
        for layer_name in target_apoz_values:
            if target_batch_count > 0:
                target_apoz_values[layer_name] /= target_batch_count

        return target_apoz_values

    def get_pruning_indices(self, apoz_values, threshold=0.93, min_channels=16):
        """
        Determine which channels to keep based on APoZ values
        Returns a dictionary mapping layer names to boolean masks
        Ensures at least min_channels channels are kept per layer
        """
        pruning_masks = {}

        for layer_idx, (layer_name, apoz) in enumerate(apoz_values.items()):
            # Use layer-dependent threshold (decreases with layer depth)
            layer_threshold = (threshold - layer_idx) / 100

            # Initially select channels with APoZ values less than threshold
            # (lower APoZ = more important channel)
            mask = apoz < layer_threshold

            # Ensure we keep at least min_channels channels
            if mask.sum() < min_channels:
                # If fewer than min_channels are selected, select the min_channels channels
                # with the lowest APoZ values (most important channels)
                indices = np.argsort(apoz)[:min_channels]
                new_mask = np.zeros_like(mask)
                new_mask[indices] = True
                mask = new_mask

            # If the layer has fewer than min_channels total channels, keep all
            if len(apoz) < min_channels:
                mask = np.ones_like(mask, dtype=bool)

            # Store the final mask for this layer
            pruning_masks[layer_name] = mask

            print(f"Layer {layer_name}: APoZ threshold={layer_threshold:.4f}, kept={mask.sum()}/{len(apoz)}")

        return pruning_masks


def apply_pruning_masks(model, pruning_masks, num_classes=10):
    """
    Create a new model with pruned channels according to masks

    Args:
        model: Original ResNet18 model
        pruning_masks: Dictionary of pruning masks for each layer
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    """
    # Create a new model to hold the pruned architecture
    pruned_model = resnet18(num_classes=num_classes)

    # Copy weights for the first layer (outside BasicBlocks)
    conv1_mask = pruning_masks['conv1']
    new_conv1 = nn.Conv2d(3, conv1_mask.sum().item(), kernel_size=3, stride=1, padding=1, bias=False)
    new_conv1.weight.data = model.conv1.weight.data[conv1_mask, :, :, :]
    pruned_model.conv1 = new_conv1

    # Copy BatchNorm weights
    new_bn1 = nn.BatchNorm2d(conv1_mask.sum().item())
    new_bn1.weight.data = model.bn1.weight.data[conv1_mask]
    new_bn1.bias.data = model.bn1.bias.data[conv1_mask]
    new_bn1.running_mean.data = model.bn1.running_mean.data[conv1_mask]
    new_bn1.running_var.data = model.bn1.running_var.data[conv1_mask]
    pruned_model.bn1 = new_bn1

    # Prune Layer 1 (may need downsample)
    in_mask = conv1_mask  # Input channels for first layer come from conv1

    for i in range(2):
        # First conv in the block
        conv1_mask = pruning_masks[f'layer1.{i}.conv1']
        # Second conv in the block
        conv2_mask = pruning_masks[f'layer1.{i}.conv2']

        # Create a new block with appropriate channels
        new_block = BasicBlock(
            in_channels=in_mask.sum().item(),
            out_channels=conv2_mask.sum().item(),
            stride=1
        )

        # Check if we need a downsample layer (if input and output channels differ)
        if in_mask.sum().item() != conv2_mask.sum().item():
            new_block.downsample = nn.Sequential(
                nn.Conv2d(in_mask.sum().item(), conv2_mask.sum().item(),
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(conv2_mask.sum().item())
            )
            # Initialize downsample weights
            # If the original block had a downsample layer, copy its weights
            if model.layer1[i].downsample is not None:
                new_block.downsample[0].weight.data = model.layer1[i].downsample[0].weight.data[conv2_mask, :]
                new_block.downsample[0].weight.data = new_block.downsample[0].weight.data[:, in_mask, :, :]

                new_block.downsample[1].weight.data = model.layer1[i].downsample[1].weight.data[conv2_mask]
                new_block.downsample[1].bias.data = model.layer1[i].downsample[1].bias.data[conv2_mask]
                new_block.downsample[1].running_mean.data = model.layer1[i].downsample[1].running_mean.data[conv2_mask]
                new_block.downsample[1].running_var.data = model.layer1[i].downsample[1].running_var.data[conv2_mask]
            else:
                # Initialize with appropriate identity mapping
                torch.nn.init.kaiming_normal_(new_block.downsample[0].weight.data)
                new_block.downsample[1].weight.data.fill_(1)
                new_block.downsample[1].bias.data.zero_()

        # Set first conv and bn weights
        new_block.conv1 = nn.Conv2d(
            in_mask.sum().item(),
            conv1_mask.sum().item(),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        temp_conv1 = model.layer1[i].conv1.weight.data[conv1_mask, :]
        new_block.conv1.weight.data = temp_conv1[:, in_mask, :, :]

        new_block.bn1 = nn.BatchNorm2d(conv1_mask.sum().item())
        new_block.bn1.weight.data = model.layer1[i].bn1.weight.data[conv1_mask]
        new_block.bn1.bias.data = model.layer1[i].bn1.bias.data[conv1_mask]
        new_block.bn1.running_mean.data = model.layer1[i].bn1.running_mean.data[conv1_mask]
        new_block.bn1.running_var.data = model.layer1[i].bn1.running_var.data[conv1_mask]

        # Set second conv and bn weights
        new_block.conv2 = nn.Conv2d(
            conv1_mask.sum().item(),
            conv2_mask.sum().item(),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        temp_conv2 = model.layer1[i].conv2.weight.data[conv2_mask, :]
        new_block.conv2.weight.data = temp_conv2[:, conv1_mask, :, :]

        new_block.bn2 = nn.BatchNorm2d(conv2_mask.sum().item())
        new_block.bn2.weight.data = model.layer1[i].bn2.weight.data[conv2_mask]
        new_block.bn2.bias.data = model.layer1[i].bn2.bias.data[conv2_mask]
        new_block.bn2.running_mean.data = model.layer1[i].bn2.running_mean.data[conv2_mask]
        new_block.bn2.running_var.data = model.layer1[i].bn2.running_var.data[conv2_mask]

        # Set the pruned block
        pruned_model.layer1[i] = new_block

        # Update the input mask for the next block
        in_mask = conv2_mask

    # Layer 2 (with downsample in first block)
    for i in range(2):
        # First conv in the block
        conv1_mask = pruning_masks[f'layer2.{i}.conv1']
        # Second conv in the block
        conv2_mask = pruning_masks[f'layer2.{i}.conv2']

        # First block has stride=2, others stride=1
        stride = 2 if i == 0 else 1

        # Create a new block
        new_block = BasicBlock(
            in_channels=in_mask.sum().item(),
            out_channels=conv2_mask.sum().item(),
            stride=stride
        )

        # Add downsample if stride > 1 or if input/output channels differ
        if stride > 1 or in_mask.sum().item() != conv2_mask.sum().item():
            new_block.downsample = nn.Sequential(
                nn.Conv2d(in_mask.sum().item(), conv2_mask.sum().item(),
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(conv2_mask.sum().item())
            )

            # Copy downsample weights if original had them
            if model.layer2[i].downsample is not None:
                temp_down = model.layer2[i].downsample[0].weight.data[conv2_mask, :]
                new_block.downsample[0].weight.data = temp_down[:, in_mask, :, :]

                new_block.downsample[1].weight.data = model.layer2[i].downsample[1].weight.data[conv2_mask]
                new_block.downsample[1].bias.data = model.layer2[i].downsample[1].bias.data[conv2_mask]
                new_block.downsample[1].running_mean.data = model.layer2[i].downsample[1].running_mean.data[conv2_mask]
                new_block.downsample[1].running_var.data = model.layer2[i].downsample[1].running_var.data[conv2_mask]
            else:
                # Initialize with appropriate identity mapping
                torch.nn.init.kaiming_normal_(new_block.downsample[0].weight.data)
                new_block.downsample[1].weight.data.fill_(1)
                new_block.downsample[1].bias.data.zero_()

        # Set first conv and bn weights
        new_block.conv1 = nn.Conv2d(
            in_mask.sum().item(),
            conv1_mask.sum().item(),
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        temp_conv1 = model.layer2[i].conv1.weight.data[conv1_mask, :]
        new_block.conv1.weight.data = temp_conv1[:, in_mask, :, :]

        new_block.bn1 = nn.BatchNorm2d(conv1_mask.sum().item())
        new_block.bn1.weight.data = model.layer2[i].bn1.weight.data[conv1_mask]
        new_block.bn1.bias.data = model.layer2[i].bn1.bias.data[conv1_mask]
        new_block.bn1.running_mean.data = model.layer2[i].bn1.running_mean.data[conv1_mask]
        new_block.bn1.running_var.data = model.layer2[i].bn1.running_var.data[conv1_mask]

        # Set second conv and bn weights
        new_block.conv2 = nn.Conv2d(
            conv1_mask.sum().item(),
            conv2_mask.sum().item(),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Fix: correctly select both output and input channels
        temp_weight = model.layer2[i].conv2.weight.data[conv2_mask, :]
        new_block.conv2.weight.data = temp_weight[:, conv1_mask, :, :]

        new_block.bn2 = nn.BatchNorm2d(conv2_mask.sum().item())
        new_block.bn2.weight.data = model.layer2[i].bn2.weight.data[conv2_mask]
        new_block.bn2.bias.data = model.layer2[i].bn2.bias.data[conv2_mask]
        new_block.bn2.running_mean.data = model.layer2[i].bn2.running_mean.data[conv2_mask]
        new_block.bn2.running_var.data = model.layer2[i].bn2.running_var.data[conv2_mask]

        # Set the pruned block
        pruned_model.layer2[i] = new_block

        # Update the input mask for the next block
        in_mask = conv2_mask

    # Layer 3 (with downsample in first block) - Similar to Layer 2
    for i in range(2):
        # First conv in the block
        conv1_mask = pruning_masks[f'layer3.{i}.conv1']
        # Second conv in the block
        conv2_mask = pruning_masks[f'layer3.{i}.conv2']

        # First block has stride=2, others stride=1
        stride = 2 if i == 0 else 1

        # Create a new block
        new_block = BasicBlock(
            in_channels=in_mask.sum().item(),
            out_channels=conv2_mask.sum().item(),
            stride=stride
        )

        # Add downsample if stride > 1 or if input/output channels differ
        if stride > 1 or in_mask.sum().item() != conv2_mask.sum().item():
            new_block.downsample = nn.Sequential(
                nn.Conv2d(in_mask.sum().item(), conv2_mask.sum().item(),
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(conv2_mask.sum().item())
            )

            # Copy downsample weights if original had them
            if model.layer3[i].downsample is not None:
                temp_down = model.layer3[i].downsample[0].weight.data[conv2_mask, :]
                new_block.downsample[0].weight.data = temp_down[:, in_mask, :, :]

                new_block.downsample[1].weight.data = model.layer3[i].downsample[1].weight.data[conv2_mask]
                new_block.downsample[1].bias.data = model.layer3[i].downsample[1].bias.data[conv2_mask]
                new_block.downsample[1].running_mean.data = model.layer3[i].downsample[1].running_mean.data[conv2_mask]
                new_block.downsample[1].running_var.data = model.layer3[i].downsample[1].running_var.data[conv2_mask]
            else:
                # Initialize with appropriate identity mapping
                torch.nn.init.kaiming_normal_(new_block.downsample[0].weight.data)
                new_block.downsample[1].weight.data.fill_(1)
                new_block.downsample[1].bias.data.zero_()

        # Set first conv and bn weights
        new_block.conv1 = nn.Conv2d(
            in_mask.sum().item(),
            conv1_mask.sum().item(),
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        temp_conv1 = model.layer3[i].conv1.weight.data[conv1_mask, :]
        new_block.conv1.weight.data = temp_conv1[:, in_mask, :, :]

        new_block.bn1 = nn.BatchNorm2d(conv1_mask.sum().item())
        new_block.bn1.weight.data = model.layer3[i].bn1.weight.data[conv1_mask]
        new_block.bn1.bias.data = model.layer3[i].bn1.weight.data[conv1_mask]
        new_block.bn1.running_mean.data = model.layer3[i].bn1.running_mean.data[conv1_mask]
        new_block.bn1.running_var.data = model.layer3[i].bn1.running_var.data[conv1_mask]

        # Set second conv and bn weights
        new_block.conv2 = nn.Conv2d(
            conv1_mask.sum().item(),
            conv2_mask.sum().item(),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        temp_conv2 = model.layer3[i].conv2.weight.data[conv2_mask, :]
        new_block.conv2.weight.data = temp_conv2[:, conv1_mask, :, :]

        new_block.bn2 = nn.BatchNorm2d(conv2_mask.sum().item())
        new_block.bn2.weight.data = model.layer3[i].bn2.weight.data[conv2_mask]
        new_block.bn2.bias.data = model.layer3[i].bn2.weight.data[conv2_mask]
        new_block.bn2.running_mean.data = model.layer3[i].bn2.running_mean.data[conv2_mask]
        new_block.bn2.running_var.data = model.layer3[i].bn2.running_var.data[conv2_mask]

        # Set the pruned block
        pruned_model.layer3[i] = new_block

        # Update the input mask for the next block
        in_mask = conv2_mask

    # Layer 4 (with downsample in first block) - Similar to Layer 3
    for i in range(2):
        # First conv in the block
        conv1_mask = pruning_masks[f'layer4.{i}.conv1']
        # Second conv in the block
        conv2_mask = pruning_masks[f'layer4.{i}.conv2']

        # First block has stride=2, others stride=1
        stride = 2 if i == 0 else 1

        # Create a new block
        new_block = BasicBlock(
            in_channels=in_mask.sum().item(),
            out_channels=conv2_mask.sum().item(),
            stride=stride
        )

        # Add downsample if stride > 1 or if input/output channels differ
        if stride > 1 or in_mask.sum().item() != conv2_mask.sum().item():
            new_block.downsample = nn.Sequential(
                nn.Conv2d(in_mask.sum().item(), conv2_mask.sum().item(),
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(conv2_mask.sum().item())
            )

            # Copy downsample weights if original had them
            if model.layer4[i].downsample is not None:
                temp_down = model.layer4[i].downsample[0].weight.data[conv2_mask, :]
                new_block.downsample[0].weight.data = temp_down[:, in_mask, :, :]

                new_block.downsample[1].weight.data = model.layer4[i].downsample[1].weight.data[conv2_mask]
                new_block.downsample[1].bias.data = model.layer4[i].downsample[1].bias.data[conv2_mask]
                new_block.downsample[1].running_mean.data = model.layer4[i].downsample[1].running_mean.data[conv2_mask]
                new_block.downsample[1].running_var.data = model.layer4[i].downsample[1].running_var.data[conv2_mask]
            else:
                # Initialize with appropriate identity mapping
                torch.nn.init.kaiming_normal_(new_block.downsample[0].weight.data)
                new_block.downsample[1].weight.data.fill_(1)
                new_block.downsample[1].bias.data.zero_()

        # Set first conv and bn weights
        new_block.conv1 = nn.Conv2d(
            in_mask.sum().item(),
            conv1_mask.sum().item(),
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        temp_conv1 = model.layer4[i].conv1.weight.data[conv1_mask, :]
        new_block.conv1.weight.data = temp_conv1[:, in_mask, :, :]

        new_block.bn1 = nn.BatchNorm2d(conv1_mask.sum().item())

        # Set second conv and bn weights
        new_block.conv2 = nn.Conv2d(
            conv1_mask.sum().item(),
            conv2_mask.sum().item(),
            kernel_size=3, stride=1, padding=1, bias=False
        )
        temp_conv2 = model.layer4[i].conv2.weight.data[conv2_mask, :]
        new_block.conv2.weight.data = temp_conv2[:, conv1_mask, :, :]

        new_block.bn2 = nn.BatchNorm2d(conv2_mask.sum().item())

        # Set the pruned block
        pruned_model.layer4[i] = new_block

        # Update the input mask for the next block
        in_mask = conv2_mask

    # Finally, adjust the fully connected layer
    fc_in_features = in_mask.sum().item()
    fc = nn.Linear(fc_in_features, num_classes)

    # Copy original FC weights (preserving pretrained weights)
    original_fc = model.fc.weight.data  # [num_classes, in_channels]
    fc.weight.data = original_fc[:, :fc_in_features].clone()
    fc.bias.data = model.fc.bias.data.clone()

    pruned_model.fc = fc

    return pruned_model


def print_model_size(model, model_name="Model"):
    """
    Print the size of the model in terms of parameters
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    print(f"{model_name} size: {size_mb:.2f} MB")
    return size_mb
