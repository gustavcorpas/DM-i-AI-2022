import tensorflow as tf
import pandas as pd
import os
import numpy as np

TRAINING_CSV_FILE = './training_data.csv'
TRAINING_IMAGE_DIR = 'training'

training_image_records = pd.read_csv(TRAINING_CSV_FILE)

train_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

train_images = []
train_targets = []
train_labels = []

for index, row in training_image_records.iterrows():
    
    (file, label, x, y, width, height) = row
    
    train_image_fullpath = os.path.join(train_image_path, str(int(file))+ '.jpg')
    print(width, height)
    train_img = tf.keras.preprocessing.image.load_img(train_image_fullpath, target_size=(800, 533))
    train_img_arr = tf.keras.preprocessing.image.img_to_array(train_img)
    
    xmin = x - width / 2
    ymin = y - height / 2
    xmax = x + width /2
    ymax = y + height / 2
    
    train_images.append(train_img_arr)
    train_targets.append((xmin, ymin, xmax, ymax))
    train_labels.append(int(label))

train_images = np.array(train_images)
train_targets = np.array(train_targets)
train_labels = np.array(train_labels)

    
print(train_images, train_targets, train_labels)