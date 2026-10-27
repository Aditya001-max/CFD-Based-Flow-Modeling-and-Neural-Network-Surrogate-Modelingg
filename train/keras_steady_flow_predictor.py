#!/usr/bin/env python3
"""
CFD Steady-State Flow Field Neural Network Predictor (U-Net Architecture)
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
Repository: https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples

Trains a deep Convolutional Encoder-Decoder (U-Net) with skip connections to predict
spatial steady-state velocity vector fields (u, v) and pressure scalar fields (p)
directly from boundary geometry representations.
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dual-compatibility imports for TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, Conv2DTranspose,
        concatenate, ZeroPadding2D, Cropping2D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    import keras
    from keras.models import Model
    from keras.layers import (
        Input, Conv2D, MaxPooling2D, Conv2DTranspose,
        concatenate, ZeroPadding2D, Cropping2D
    )
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from vtm_data import VTKDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SteadyFlowPredictor")


def pad_to_match(target_tensor, source_tensor):
    """Pads source_tensor with ZeroPadding2D to match the spatial dimensions of target_tensor."""
    # When shapes are known statically
    target_shape = target_tensor.shape
    source_shape = source_tensor.shape
    if target_shape[1] is not None and source_shape[1] is not None:
        diff_h = target_shape[1] - source_shape[1]
        diff_w = target_shape[2] - source_shape[2]
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        if diff_h > 0 or diff_w > 0:
            return ZeroPadding2D(((pad_top, pad_bottom), (pad_left, pad_right)))(source_tensor)
        elif diff_h < 0 or diff_w < 0:
            crop_top = abs(diff_h) // 2
            crop_bottom = abs(diff_h) - crop_top
            crop_left = abs(diff_w) // 2
            crop_right = abs(diff_w) - crop_left
            return Cropping2D(((crop_top, crop_bottom), (crop_left, crop_right)))(source_tensor)
    return source_tensor


def build_flow_unet(input_shape):
    """
    Constructs an Encoder-Decoder U-Net for spatial velocity and pressure field estimation.
    Output channels: 3 (Index 0: u-velocity, Index 1: v-velocity, Index 2: pressure).
    """
    inputs = Input(shape=input_shape, name="flow_geometry_input")

    # Encoder Stage 1
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1_1")(inputs)
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1_2")(c1)
    p1 = MaxPooling2D(pool_size=(2, 2), name="enc_pool1")(c1)

    # Encoder Stage 2
    c2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2_1")(p1)
    c2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2_2")(c2)
    p2 = MaxPooling2D(pool_size=(2, 2), name="enc_pool2")(c2)

    # Encoder Stage 3
    c3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3_1")(p2)
    c3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3_2")(c3)
    p3 = MaxPooling2D(pool_size=(2, 2), name="enc_pool3")(c3)

    # Encoder Stage 4
    c4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="enc_conv4_1")(p3)
    c4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="enc_conv4_2")(c4)
    p4 = MaxPooling2D(pool_size=(2, 2), name="enc_pool4")(c4)

    # Bottleneck
    bn = Conv2D(512, (3, 3), activation="relu", padding="same", name="bottleneck_1")(p4)
    bn = Conv2D(512, (3, 3), activation="relu", padding="same", name="bottleneck_2")(bn)

    # Decoder Stage 4
    u4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", name="up_conv4")(bn)
    u4_padded = pad_to_match(c4, u4)
    merge4 = concatenate([u4_padded, c4], axis=3, name="skip_merge4")
    d4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="dec_conv4_1")(merge4)
    d4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="dec_conv4_2")(d4)

    # Decoder Stage 3
    u3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="up_conv3")(d4)
    u3_padded = pad_to_match(c3, u3)
    merge3 = concatenate([u3_padded, c3], axis=3, name="skip_merge3")
    d3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_conv3_1")(merge3)
    d3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_conv3_2")(d3)

    # Decoder Stage 2
    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="up_conv2")(d3)
    u2_padded = pad_to_match(c2, u2)
    merge2 = concatenate([u2_padded, c2], axis=3, name="skip_merge2")
    d2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv2_1")(merge2)
    d2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv2_2")(d2)

    # Decoder Stage 1
    u1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", name="up_conv1")(d2)
    u1_padded = pad_to_match(c1, u1)
    merge1 = concatenate([u1_padded, c1], axis=3, name="skip_merge1")
    d1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv1_1")(merge1)
    d1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv1_2")(d1)

    # 1x1 Conv output for 3 channels (u, v, p)
    outputs = Conv2D(3, (1, 1), activation="linear", name="flow_and_pressure_output")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="cfd_flow_unet")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFD Steady Flow & Pressure Spatial Surrogate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam optimizer learning rate (default: 1e-4)")
    parser.add_argument("--data-dir", type=str, default="../data", help="Path to simulation data root")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset if true")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of synthetic samples (default: 100)")
    parser.add_argument("--model-path", type=str, default="steady_flow_predictor_model.keras", help="Model checkpoint path")
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
    train_flows = dataset.steady_flows[:dataset.split_line]

    test_geoms = dataset.geometries[dataset.split_line:]
    test_flows = dataset.steady_flows[dataset.split_line:]

    x_train = np.stack(train_geoms, axis=0)
    y_train = np.stack(train_flows, axis=0)
    x_test = np.stack(test_geoms, axis=0)
    y_test = np.stack(test_flows, axis=0)

    logger.info(f"Input geometry dimensions: {x_train.shape[1:]}")
    logger.info(f"Flow field output dimensions: {y_train.shape[1:]}")
    logger.info(f"Training partition: {x_train.shape[0]} samples | Test partition: {x_test.shape[0]} samples")

    model = build_flow_unet(x_train.shape[1:])
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

    # Plot visual field comparisons (Velocity magnitude and Pressure)
    num_plots = min(3, len(predictions))
    for i in range(num_plots):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))

        # Velocity magnitude
        pred_u = predictions[i, :, :, 0]
        true_u = y_test[i, :, :, 0]
        geom = x_test[i, :, :, 0]

        axes[0].imshow(pred_u, cmap="jet")
        axes[0].set_title(f"Sample {i + 1} | Predicted Velocity (u-component)")
        axes[0].axis("off")

        axes[1].imshow(true_u, cmap="jet")
        axes[1].set_title(f"Sample {i + 1} | Ground Truth Velocity (u-component)")
        axes[1].axis("off")

        axes[2].imshow(geom, cmap="gray")
        axes[2].set_title(f"Sample {i + 1} | Channel Boundary Mask")
        axes[2].axis("off")

        plot_path = os.path.join(args.plot_dir, f"flow_eval_sample_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved evaluation artifact to {plot_path}")

    logger.info("Steady flow field surrogate training completed successfully.")


if __name__ == "__main__":
    main()
