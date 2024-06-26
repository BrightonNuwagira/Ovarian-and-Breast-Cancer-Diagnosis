import os
import pandas as pd
import numpy as np
from PIL import Image
import SimpleITK as sitk
from PIL import Image
import os
# Define the path to the dataset
busi_dataset_path = '/scratch/09457/bxb210001/Ovarian_Cancer'


train_images_path = '/scratch/09457/bxb210001/Ovarian_Cancer/Train_Images'
test_images_path =  '/scratch/09457/bxb210001/Ovarian_Cancer/Test_Images'

# Function to load images and labels
def load_images_and_labels(base_path):
    image_paths = []
    labels = []
    for category in ['CC', 'EC', 'HGSC', 'LGSC', 'MC']:
        category_path = os.path.join(base_path, category)
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith(('.pgm','.png', '.PNG', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
                    labels.append(category)
    return image_paths, labels

train_image_paths, train_labels = load_images_and_labels(train_images_path)
# Load test images and labels
test_image_paths, test_labels = load_images_and_labels(test_images_path)


df_train_images = pd.DataFrame({'image': train_image_paths, 'Label': train_labels})
df_test_images = pd.DataFrame({'image': test_image_paths, 'Label': test_labels})

print("Total Number of Train Images:", len(df_train_images))
print("Total Number of Test Images:", len(df_test_images))

# Convert labels to binary format (assuming binary classification)
label_mapping = { 'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4 }

df_train_images['Label'] = df_train_images['Label'].map(label_mapping)
df_test_images['Label'] = df_test_images['Label'].map(label_mapping)




#BREAKHIS DATA LOAD i.e  40X,100X, 200X, 400X



busi_dataset_path = '/scratch/09457/bxb210001/BREAKHIS/40X'


image_lists = {}
for folder_name in ['benign', 'malignant']:
    image_paths = []
    folder_path = os.path.join(busi_dataset_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        continue
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    image_lists[folder_name] = image_paths
    print(f"Found {len(image_paths)} images in folder {folder_name}")

all_images = image_lists.get('benign', []) + image_lists.get('malignant', [])
print("Total Number of Images:", len(all_images))

labels = [0] * len(image_lists.get('benign', [])) + [1] * len(image_lists.get('malignant', []))
df_image = pd.DataFrame({'image': all_images, 'Label': labels})
print(df_image)


