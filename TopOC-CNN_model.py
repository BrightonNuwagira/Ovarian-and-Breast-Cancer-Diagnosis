
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score
import concurrent.futures
from tensorflow.keras.applications import ResNet101, ResNet50, DenseNet121, VGG16, EfficientNetB0, InceptionV3, MobileNetV2, InceptionResNetV2
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout, Input
from PIL import Image
import time




def create_preprocessing(model_name):
    if model_name == "ResNet101":
        return ResNet101, preprocess_resnet
    elif model_name == "ResNet50":
        return ResNet50, preprocess_resnet
    elif model_name == "DenseNet121":
        return DenseNet121, preprocess_densenet
    elif model_name == "VGG16":
        return VGG16, preprocess_vgg
    elif model_name == "EfficientNetB0":
        return EfficientNetB0, preprocess_efficientnet
    elif model_name == "InceptionV3":
        return InceptionV3, preprocess_inceptionv3
    elif model_name == "MobileNetV2":
        return MobileNetV2, preprocess_mobilenetv2
    elif model_name == "InceptionResNetV2":
        return InceptionResNetV2, preprocess_inceptionresnetv2
    else:
        raise ValueError("Unsupported model name")



def preprocess_image(img_path, model_name):
    _, preprocess_function = create_preprocessing(model_name)
    img = Image.open(img_path)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    preprocessed_img = preprocess_function(img_array)
    return preprocessed_img


def create_MLP(dim):
    model = Sequential()
    model.add(Dense(256, input_dim=dim, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    return model


def create_model(model_name, tda_num_features, n_classes):
    base_model_class, preprocess_function = create_preprocessing(model_name)
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    mlp = create_MLP(tda_num_features)

    cnn_input = Input(shape=(224, 224, 3), name='cnn_input')
    x = base_model(cnn_input)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    
    mlp_input = Input(shape=(tda_num_features,), name='mlp_input')
    mlp_output = mlp(mlp_input)

    combinedInput = concatenate([mlp_output, x])
    x = Dense(256, activation="relu")(combinedInput)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[mlp_input, cnn_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
       metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    return model




def train_and_evaluate_single(num_epochs, model_name, tda_num_features, n_classes, 
                              train_data_array_images, y_train, 
                              test_data_array_images, y_test, 
                              X_tda_train, y_tda_train, 
                              X_tda_test, y_tda_test):
    model = create_model(model_name, tda_num_features, n_classes)
    print(f"Training with {num_epochs} epochs for model {model_name}...")

    model.fit(x=[X_tda_train, train_data_array_images], y=y_tda_train,
              validation_data=([X_tda_test, test_data_array_images], y_tda_test),
              epochs=num_epochs, batch_size=64)

    y_pred = model.predict([X_tda_test, test_data_array_images])
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_tda_test, axis=1)

    precision = precision_score(y_test_classes, y_pred_classes, average='macro')
    recall = recall_score(y_test_classes, y_pred_classes, average='macro')
    auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')
    balanced_acc = balanced_accuracy_score(y_test_classes, y_pred_classes)

    metrics = {
        'Model': model_name,
        'Epochs': num_epochs,
        'Balanced Accuracy': balanced_acc,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc,
    }
    return metrics


def train_and_evaluate_parallel(model_name, epochs_list, train_data_array_images, y_train, 
                                test_data_array_images, y_test, X_tda_train, y_tda_train, 
                                X_tda_test, y_tda_test):
    results_list = []
    tda_num_features = X_tda_train.shape[1]
    n_classes = y_tda_train.shape[1]  

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for num_epochs in epochs_list:
            futures.append(executor.submit(train_and_evaluate_single, num_epochs, model_name, tda_num_features, n_classes, 
                                           train_data_array_images, y_train, 
                                           test_data_array_images, y_test, 
                                           X_tda_train, y_tda_train, 
                                           X_tda_test, y_tda_test))
        concurrent.futures.wait(futures)
        results_list.extend([future.result() for future in futures])
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'OvarianCancer_{model_name}_128_BF_CNN.csv', index=False)
    print(f"Evaluation Results for {model_name} with Different Epochs:")
    print(results_df)


model_names = ['VGG16','DenseNet121','EfficientNetB0']  
epochs_list = [100]  

df_train_features = pd.read_excel('train_all_chanel_with_label_ovarian_project.xlsx')  
df_test_features = pd.read_excel('test_all_chanel_with_label_ovarian_project.xlsx') 



def reindex_columns(df):
    current_columns = list(df.columns)
    last_column = current_columns[-1]
    new_indices = list(range(len(current_columns) - 1))
    new_indices.append(last_column)
    df.columns = new_indices
    return df


df_train_features = reindex_columns(df_train_features)
df_test_features = reindex_columns(df_test_features)


for model_name in model_names:
    print(f"Processing model: {model_name}")

 
    train_images_Datta = df_train_images['image'].tolist()
    train_data_array = []
    for img_path in train_images_Datta:
        img_data = preprocess_image(img_path, model_name)
        train_data_array.append(img_data)
    train_data_array_images = np.array(train_data_array)
    y_train = df_train_images['Label'].to_numpy()

   
    test_images_Datta = df_test_images['image'].tolist()
    test_data_array = []
    for img_path in test_images_Datta:
        img_data = preprocess_image(img_path, model_name)
        test_data_array.append(img_data)
    test_data_array_images = np.array(test_data_array)
    y_test = df_test_images['Label'].to_numpy()

    
    X_tda_train = df_train_features.iloc[:, :-1].to_numpy()
    y_tda_train = df_train_features.iloc[:, -1].to_numpy()
    X_tda_test = df_test_features.iloc[:, :-1].to_numpy()
    y_tda_test = df_test_features.iloc[:, -1].to_numpy()

   
    label_encoder = LabelEncoder()
    y_tda_train_encoded = label_encoder.fit_transform(y_tda_train)
    y_tda_train_categorical = tf.keras.utils.to_categorical(y_tda_train_encoded)

    y_tda_test_encoded = label_encoder.transform(y_tda_test)
    y_tda_test_categorical = tf.keras.utils.to_categorical(y_tda_test_encoded)

   
    train_and_evaluate_parallel(model_name, epochs_list, 
                                train_data_array_images, y_tda_train_categorical, 
                                test_data_array_images, y_tda_test_categorical, 
                                X_tda_train, y_tda_train_categorical,
                                X_tda_test, y_tda_test_categorical)

    print(f"Completed processing for model: {model_name}")

