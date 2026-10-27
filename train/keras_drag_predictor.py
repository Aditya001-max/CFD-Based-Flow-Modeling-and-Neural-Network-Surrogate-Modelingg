#!/usr/bin/env python3
"""
CFD Drag Force Neural Network Predictor
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
Repository: https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples

Trains a Deep Convolutional Neural Network (CNN) to predict steady-state aerodynamic drag force
directly from 2D flow boundary raster representations.
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless & automated pipelines
import matplotlib.pyplot as plt

# Dual-compatibility imports for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    import keras
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from vtm_data import VTKDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DragPredictor")


def build_drag_predictor(input_shape):
    """
    Constructs the 4-stage convolutional feature extractor with dense regression head.
    Architecture:
      Conv(4, 3x3) -> Conv(4, 3x3) -> MaxPool(2x2)
      Conv(8, 3x3) -> Conv(8, 3x3) -> MaxPool(2x2)
      Conv(16, 3x3) -> Conv(16, 3x3) -> MaxPool(2x2)
      Conv(64, 3x3) -> Conv(64, 3x3) -> MaxPool(2x2)
      Flatten -> Dense(512, ReLU) -> Dropout(0.5) -> Dense(1, Linear)
    """
    inputs = Input(shape=input_shape, name="geometry_input")

    # Stage 1
    x = Conv2D(4, (3, 3), activation="relu", padding="same", name="conv1_1")(inputs)
    x = Conv2D(4, (3, 3), activation="relu", padding="same", name="conv1_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Stage 2
    x = Conv2D(8, (3, 3), activation="relu", padding="same", name="conv2_1")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same", name="conv2_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Stage 3
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv3_1")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="conv3_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Stage 4
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4_1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Fully connected regression head
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu", name="fc1")(x)
    x = Dropout(0.5, name="dropout")(x)
    outputs = Dense(1, activation="linear", name="predicted_drag")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cfd_drag_cnn")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFD Aerodynamic Drag Surrogate Model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam optimizer learning rate (default: 1e-4)")
    parser.add_argument("--data-dir", type=str, default="../data", help="Path to simulation data root")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset if true")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of synthetic samples (default: 200)")
    parser.add_argument("--model-path", type=str, default="drag_predictor_model.keras", help="Model checkpoint path")
    parser.add_argument("--plot-dir", type=str, default="../figs", help="Directory to save evaluation plots")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    dataset = VTKDataset(base_dir=args.data_dir)
    loaded = dataset.load_data()

    if loaded == 0 or args.synthetic:
        logger.info(f"No simulation files found or --synthetic flag set. Generating {args.num_samples} samples.")
        dataset.generate_synthetic_data(num_samples=args.num_samples)

    train_geoms = dataset.geometries[:dataset.split_line]
    train_drags = [val[-1:] for val in dataset.drag_vectors[:dataset.split_line]]

    test_geoms = dataset.geometries[dataset.split_line:]
    test_drags = [val[-1:] for val in dataset.drag_vectors[dataset.split_line:]]

    x_train = np.stack(train_geoms, axis=0)
    y_train = np.stack(train_drags, axis=0)
    x_test = np.stack(test_geoms, axis=0)
    y_test = np.stack(test_drags, axis=0)

    logger.info(f"Input geometry dimensions: {x_train.shape[1:]}")
    logger.info(f"Training partition: {x_train.shape[0]} samples | Test partition: {x_test.shape[0]} samples")

    model = build_drag_predictor(x_train.shape[1:])
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=args.lr),
        metrics=["mean_squared_error", "mean_absolute_error"]
    )
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(args.model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    eval_results = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test Loss (MSE): {eval_results[0]:.4f} | Test MAE: {eval_results[1]:.4f}")

    predictions = model.predict(x_test, batch_size=args.batch_size)

    # Plot sample comparisons
    num_plots = min(3, len(predictions))
    for i in range(num_plots):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(x_test[i, :, :, 0], cmap="Blues_r")
        ax.set_title(
            f"Sample {i + 1} | Ground Truth Drag: {float(y_test[i]):.4f} | Predicted Drag: {float(predictions[i]):.4f}"
        )
        ax.axis("off")
        plot_path = os.path.join(args.plot_dir, f"drag_eval_sample_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved evaluation artifact to {plot_path}")

    logger.info("Drag prediction surrogate training completed successfully.")


if __name__ == "__main__":
    main()
