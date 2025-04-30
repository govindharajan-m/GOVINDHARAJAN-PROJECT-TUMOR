import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import cv2
import streamlit as st
from sklearn.model_selection import train_test_split
import zipfile

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "tumor_segmentation_model.keras"

class TumorSegmenter:
    def __init__(self):
        self.model = None
        self.sample_images = self._generate_sample_data()
        
    def _generate_sample_data(self):
        """Generate synthetic sample data to avoid file path dependencies"""
        images = []
        masks = []
        for i in range(10):  # 10 sample images
            # Create synthetic MRI-like image (random noise with central "tumor")
            img = np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 0.5
            center = (IMG_SIZE//2, IMG_SIZE//2)
            radius = IMG_SIZE//4
            y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] += 0.5  # Simulate tumor
            
            # Create corresponding mask
            mask = mask.astype(float).reshape(IMG_SIZE, IMG_SIZE, 1)
            
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.array(masks)
    
    def load_data(self):
        """Load or generate sample data"""
        images, masks = self.sample_images
        masked_images = images * masks
        X = np.concatenate((images, masked_images), axis=-1)
        return train_test_split(X, masks, test_size=0.2, random_state=42)
    
    def build_model(self):
        """Create modified U-Net model"""
        inputs = Input((IMG_SIZE, IMG_SIZE, 6))
        
        # Encoder
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        
        # Decoder
        up1 = UpSampling2D((2, 2))(conv3)
        concat1 = Concatenate()([conv2, up1])
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
        up2 = UpSampling2D((2, 2))(conv4)
        concat2 = Concatenate()([conv1, up2])
        conv5 = Conv2D(1, (1, 1), activation='sigmoid')(concat2)
        
        model = Model(inputs=inputs, outputs=conv5)
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self):
        """Train the model"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        self.model = self.build_model()
        
        # Train on GPU if available
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        with tf.device(device):
            history = self.model.fit(X_train, y_train,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  validation_data=(X_test, y_test))
        
        # Save model in modern Keras format
        self.model.save(MODEL_NAME)
        return history
    
    def predict(self, image_array):
        """Make predictions on new images"""