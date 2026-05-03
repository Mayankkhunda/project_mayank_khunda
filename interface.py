# Model
from model import FlowerCNN as TheModel

# Training function
from train import train_model as the_trainer

# Prediction function
from predict import predict_images as the_predictor

# Dataset and DataLoader
from dataset import FlowerDataset as TheDataset
from dataset import flowerLoader as the_dataloader

# Config parameters
from config import batchsize as the_batch_size
from config import epochs as total_epochs
