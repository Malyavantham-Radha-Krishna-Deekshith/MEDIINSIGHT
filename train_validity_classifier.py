import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ---------------------
# Config
# ---------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = os.getenv('DATA_DIR', "validator"
)# Use environment variable or default path

# ---------------------
# Validate Dataset Path
# ---------------------
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset path '{DATA_DIR}' does not exist.")
if not os.listdir(DATA_DIR):
    raise FileNotFoundError(f"Dataset path '{DATA_DIR}' is empty.")
print(f"✅ Dataset path '{DATA_DIR}' is valid.")

# ---------------------
# Data Generators
# ---------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ---------------------
# Model Architecture
# ---------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ---------------------
# Compile Model
# ---------------------
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------
# Callbacks
# ---------------------
checkpoint = ModelCheckpoint(
    "best_validity_classifier.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ---------------------
# Train Model
# ---------------------
print("🚀 Starting training...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping]
)

# ---------------------
# Save Final Model
# ---------------------
model.save("validity_classifier.h5")
print("✅ Model saved as validity_classifier.h5")