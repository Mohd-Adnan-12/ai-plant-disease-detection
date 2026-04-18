"""
============================================================
Smart Plant Disease Detection System
train.py — Enhanced Training Script
============================================================
Upgrades over original notebook:
  - Transfer learning with MobileNetV2 (pretrained on ImageNet)
  - Fine-tuning with unfreezing top layers
  - Learning rate scheduling + early stopping
  - Dropout regularization to reduce overfitting
  - Saves training history as JSON for dashboard use
  - Exports confusion matrix and accuracy/loss plots
  - Saves class label mapping automatically
============================================================
Dataset: Healthy / Powdery / Rust (3-class leaf disease)
Compatible with PlantVillage-style folder structure
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)     # MobileNetV2 standard input
BATCH_SIZE  = 32
EPOCHS_HEAD = 10             # Phase 1: train classifier head only
EPOCHS_FINE = 10             # Phase 2: fine-tune top MobileNetV2 layers
BASE_LR     = 1e-3
FINE_LR     = 1e-5
DROPOUT     = 0.4

TRAIN_DIR = "Dataset/Train/Train"
VAL_DIR   = "Dataset/Validation/Validation"
TEST_DIR  = "Dataset/Test/Test"

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ── Data Generators ─────────────────────────────────────────────────────────
print("📂 Loading datasets...")

# Augmentation only for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Only rescale for val/test
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class label mapping so predict.py can use it
class_indices = train_gen.class_indices  # e.g. {"Healthy":0, "Powdery":1, "Rust":2}
label_map = {v: k for k, v in class_indices.items()}  # {0:"Healthy", ...}

with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
    json.dump(label_map, f)
print(f"✅ Class labels saved: {label_map}")

# ── Build Model ─────────────────────────────────────────────────────────────
print("\n🔧 Building MobileNetV2 transfer learning model...")

# Load MobileNetV2 pretrained on ImageNet, exclude top classifier
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Phase 1: Freeze all base layers, train only the new head
base_model.trainable = False

# Custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(DROPOUT)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(DROPOUT / 2)(x)
outputs = Dense(len(class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=BASE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Total params: {model.count_params():,}")
print(f"Trainable params (head only): {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

# ── Phase 1: Train Head ─────────────────────────────────────────────────────
print(f"\n🚀 Phase 1: Training classifier head ({EPOCHS_HEAD} epochs)...")

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-7),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_phase1.keras"),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    verbose=1
)

# ── Phase 2: Fine-tune ───────────────────────────────────────────────────────
print(f"\n🔓 Phase 2: Fine-tuning top MobileNetV2 layers ({EPOCHS_FINE} epochs)...")

# Unfreeze the top 30 layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use a much smaller LR to avoid destroying pretrained weights
model.compile(
    optimizer=Adam(learning_rate=FINE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-8),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_final.keras"),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    verbose=1,
    initial_epoch=len(history1.history['accuracy'])  # continue epoch count
)

# ── Merge Histories ──────────────────────────────────────────────────────────
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

full_history = merge_histories(history1, history2)

# Save history for dashboard display
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(full_history, f)
print("✅ Training history saved.")

# ── Save Final Model ─────────────────────────────────────────────────────────
model.save(os.path.join(OUTPUT_DIR, "plant_disease_model.keras"))
model.save(os.path.join(OUTPUT_DIR, "plant_disease_model.h5"))  # Legacy format
print("✅ Model saved in both .keras and .h5 formats.")

# ── Plot Accuracy & Loss ─────────────────────────────────────────────────────
epochs = range(1, len(full_history['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Training Performance", fontsize=16, fontweight='bold')

# Accuracy
axes[0].plot(epochs, full_history['accuracy'], 'b-o', label='Train', markersize=4)
axes[0].plot(epochs, full_history['val_accuracy'], 'r-o', label='Validation', markersize=4)
axes[0].axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs, full_history['loss'], 'b-o', label='Train', markersize=4)
axes[1].plot(epochs, full_history['val_loss'], 'r-o', label='Validation', markersize=4)
axes[1].axvline(x=EPOCHS_HEAD, color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Training curves saved → assets/training_curves.png")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
print("\n📊 Generating confusion matrix on test set...")

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

class_names = list(class_indices.keys())
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names, ax=ax
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved → assets/confusion_matrix.png")

# ── Classification Report ─────────────────────────────────────────────────────
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n📋 Classification Report:\n")
print(report)

# Save report to text file
with open("assets/classification_report.txt", "w") as f:
    f.write(report)

print("\n✅ Training complete! All artifacts saved to models/ and assets/")