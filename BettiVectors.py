import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pandas as pd
from gtda.images import Binarizer
from gtda.images import SignedDistanceFiltration
from gtda.images import ErosionFiltration
from gtda.homology import CubicalPersistence
from PIL import Image
import matplotlib.pyplot as plt
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram
from joblib import Parallel, delayed
import os




homology_dimensions = [0, 1]
CP = CubicalPersistence(homology_dimensions=homology_dimensions, coeff=3, n_jobs=-1)
BC = BettiCurve(n_bins=50)

data = []


def process_channel(channel_data):
    diagram = CP.fit_transform(np.array(channel_data)[None, :, :])
    betti_curves = BC.fit_transform(diagram)
    return np.round(np.reshape(betti_curves, 100), 0)

for image_path inimages_Datta:
    image = Image.open(image_path)

    # Convert to grayscale and process
    gray_image = image.convert('L')
    im_gray_h1 = np.array(gray_image)
    gray_features = process_channel(im_gray_h1)

    # Extract and process color channels
    image_array = np.array(image)
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    red_features = process_channel(red_channel)
    green_features = process_channel(green_channel)
    blue_features = process_channel(blue_channel)


    combined_features = np.concatenate([gray_features, red_features, green_features, blue_features])
    data.append(combined_features)

df = pd.DataFrame(data)
results_df = pd.concat([df, df_train_images['Label'].reset_index(drop=True)], axis=1)


print(results_df)

