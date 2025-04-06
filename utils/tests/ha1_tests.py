import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def test_number_of_samples(dataset, expected_samples, dataset_class):
    """Test if the dataset has the correct number of samples."""
    actual_samples = len(dataset)
    assert actual_samples == expected_samples, \
        f"Expected {expected_samples} samples, got {actual_samples} samples"

def test_label(dataset, expected_label, dataset_class):
    """Test if the dataset returns correct label format."""
    sample, label = dataset[0]
    assert isinstance(label, (int, torch.Tensor)), \
        f"Label should be int or tensor, got {type(label)}"
    if isinstance(label, torch.Tensor):
        assert label.dim() == 0, \
            f"Label tensor should be scalar, got shape {label.shape}"

def test_dataloader(dataloader):
    """Test if the dataloader returns correct batch format."""
    batch = next(iter(dataloader))
    assert isinstance(batch, (list, tuple)), \
        f"Batch should be list or tuple, got {type(batch)}"
    assert len(batch) == 2, \
        f"Batch should contain 2 elements (images, labels), got {len(batch)}"
    
    images, labels = batch
    assert isinstance(images, torch.Tensor), \
        f"Images should be tensor, got {type(images)}"
    assert isinstance(labels, torch.Tensor), \
        f"Labels should be tensor, got {type(labels)}"

def test_model(model):
    """Test if the model has the correct structure."""
    assert isinstance(model, nn.Module), \
        f"Model should be nn.Module, got {type(model)}"
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        output = model(dummy_input)
        assert isinstance(output, torch.Tensor), \
            f"Model output should be tensor, got {type(output)}"
    except Exception as e:
        raise AssertionError(f"Model forward pass failed: {str(e)}")

def test_architecture(model):
    """Test if the model has the correct architecture."""
    # Check if model has required layers
    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
    has_pool = any(isinstance(m, nn.MaxPool2d) for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    
    assert has_conv, "Model should have convolutional layers"
    assert has_pool, "Model should have pooling layers"
    assert has_fc, "Model should have fully connected layers"

def test_output_to_label(output_to_label):
    """Test if the output_to_label function works correctly."""
    # Test with dummy output
    dummy_output = torch.tensor([0.6, 0.4])
    try:
        label = output_to_label(dummy_output)
        assert isinstance(label, (int, torch.Tensor)), \
            f"Label should be int or tensor, got {type(label)}"
    except Exception as e:
        raise AssertionError(f"output_to_label function failed: {str(e)}")

def test_transfer_learning_head(head):
    """Test if the transfer learning head has correct structure."""
    assert isinstance(head, nn.Module), \
        f"Head should be nn.Module, got {type(head)}"
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 512)  # Assuming input size of 512
    try:
        output = head(dummy_input)
        assert isinstance(output, torch.Tensor), \
            f"Head output should be tensor, got {type(output)}"
    except Exception as e:
        raise AssertionError(f"Head forward pass failed: {str(e)}")

def test_vgg_model_1(vgg_model, head):
    """Test if VGG model and head are properly connected."""
    # Test forward pass through both model and head
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        features = vgg_model(dummy_input)
        output = head(features)
        assert isinstance(output, torch.Tensor), \
            f"Combined model output should be tensor, got {type(output)}"
    except Exception as e:
        raise AssertionError(f"Combined model forward pass failed: {str(e)}")

def test_vgg_model_2(vgg_model, head):
    """Test if VGG model parameters are frozen."""
    for param in vgg_model.parameters():
        assert not param.requires_grad, \
            "VGG model parameters should be frozen (requires_grad=False)"

def test_vgg_model_parameters_for_transfer_learning(vgg_model):
    """Test if VGG model parameters are properly set for transfer learning."""
    for param in vgg_model.parameters():
        assert not param.requires_grad, \
            "VGG model parameters should be frozen for transfer learning"

def test_dataloader_for_transfer_learning(dataloader):
    """Test if dataloader is properly configured for transfer learning."""
    batch = next(iter(dataloader))
    images, labels = batch
    
    # Check image normalization
    assert images.min() >= 0 and images.max() <= 1, \
        "Images should be normalized to [0,1] range"
    
    # Check image size
    assert images.shape[2:] == (224, 224), \
        f"Images should be 224x224, got {images.shape[2:]}"

def test_vgg_model_parameters_for_fine_tuning(vgg_model):
    """Test if VGG model parameters are properly set for fine-tuning."""
    # Check if some parameters are trainable
    trainable_params = sum(p.requires_grad for p in vgg_model.parameters())
    assert trainable_params > 0, \
        "Some VGG model parameters should be trainable for fine-tuning"

def test_learning_rate(learning_rate):
    """Test if learning rate is within reasonable range."""
    assert isinstance(learning_rate, float), \
        f"Learning rate should be float, got {type(learning_rate)}"
    assert 1e-6 <= learning_rate <= 1e-1, \
        f"Learning rate {learning_rate} is outside reasonable range [1e-6, 1e-1]"

def test_dataloaders_for_final_training(train_dataloader, val_dataloader, verbose=False):
    """Test if dataloaders are properly configured for final training."""
    # Test training dataloader
    train_batch = next(iter(train_dataloader))
    images, labels = train_batch
    
    if verbose:
        print(f"Training batch shape: {images.shape}")
        print(f"Training labels shape: {labels.shape}")
    
    # Test validation dataloader
    val_batch = next(iter(val_dataloader))
    val_images, val_labels = val_batch
    
    if verbose:
        print(f"Validation batch shape: {val_images.shape}")
        print(f"Validation labels shape: {val_labels.shape}")
    
    # Check if shapes match between train and val
    assert images.shape[1:] == val_images.shape[1:], \
        "Train and validation images should have same shape"
    assert labels.shape[1:] == val_labels.shape[1:], \
        "Train and validation labels should have same shape" 