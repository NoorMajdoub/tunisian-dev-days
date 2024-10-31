#------------------------------------------
#Step 1 imports and Paths defintions and data loading
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNet, InceptionV3, ResNet50
import tensorflow as tf
train_data_path="caridacimages/train/train/"
test_data_path="caridacimages/test/test/"


# Loading the training data
filepaths = []
labels = []

# Reading images from the training dataset
folds = os.listdir(train_data_path) #two folds ['false', 'true']
for fold in folds:
    f_path = os.path.join(train_data_path, fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path, file))
        labels.append(fold)

#------------------------------------------
# Step2 Creating DataFrame for training data
train_df = pd.DataFrame({
    'filepaths': filepaths,
    'label': labels
})

test_filepaths = []
test_labels = []

test_folds = os.listdir(test_data_path)
for fold in test_folds:
    test_f_path = os.path.join(test_data_path, fold)
    test_filelists = os.listdir(test_f_path)

    for file in test_filelists:
        test_filepaths.append(os.path.join(test_f_path, file))
        test_labels.append(fold)

# Creating DataFrame for test data
test_df = pd.DataFrame({
    'filepaths': test_filepaths,
    'label': test_labels
})
#------------------------------------------
# Step3 Imagedata preprocessing

# change images size
img_width, img_height = 224, 224

#   data augmentation with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Training and validation generators
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

valid_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
#---------------------------------------------------
#Step 4 model architecture definiton
# Define L2 regularizer globally
regularizer = tf.keras.regularizers.l2(0.01)


# Function to create a model (MobileNet, InceptionV3, ResNet50) with simplified architecture
def build_model(base_model):
    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(0.4)(x)  # Reduced dropout to avoid over-regularization
    x = Dense(128, activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)


# Ensemble function to apply weighted average to the outputs
def weighted_ensemble(models, model_input, weights):
    outputs = [model(model_input) for model in models]
    weighted_outputs = [outputs[i] * weights[i] for i in range(len(models))]
    y = tf.keras.layers.add(weighted_outputs)
    return Model(model_input, y, name='weighted_ensemble')


# Define models as before
mobilenet_base = MobileNet(include_top=False, input_shape=(img_width, img_height, 3), weights='imagenet')
mobilenet_model = build_model(mobilenet_base)

inception_base = InceptionV3(include_top=False, input_shape=(img_width, img_height, 3), weights='imagenet')
inception_model = build_model(inception_base)

resnet_base = ResNet50(include_top=False, input_shape=(img_width, img_height, 3), weights='imagenet')
resnet_model = build_model(resnet_base)

# Input layer (common to all models)
model_input = Input(shape=(img_width, img_height, 3))

# Define weights for each model (you can adjust these)
model_weights = [0.4, 0.4, 0.2]

# Build the weighted ensemble model
ensemble_model = weighted_ensemble([mobilenet_model, inception_model, resnet_model], model_input, model_weights)

# Compile the ensemble model with L2 penalty in the loss function
ensemble_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Summary of the ensemble model
ensemble_model.summary()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_weighted_ensemble_model.keras', monitor='val_accuracy', save_best_only=True)
callbacks = [early_stopping, checkpoint]

# Training and cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"Training fold {fold + 1}...")

    # Create training and validation generators for the fold
    train_gen_fold = train_datagen.flow_from_dataframe(
        dataframe=train_df.iloc[train_idx],
        x_col='filepaths',
        y_col='label',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
    )

    val_gen_fold = train_datagen.flow_from_dataframe(
        dataframe=train_df.iloc[val_idx],
        x_col='filepaths',
        y_col='label',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
    )

    # Train the model
    history = ensemble_model.fit(
        train_gen_fold,
        validation_data=val_gen_fold,
        epochs=6,
        callbacks=callbacks,
        verbose=1
    )

# Evaluate on the test set
test_loss, test_acc = ensemble_model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
ensemble_model.save('weighted_ensemble_model.h5')  # Save to HDF5 format