"""
Unit Tests for Deep Learning Architectures (CNN Drag Predictor & U-Net Flow Estimator)
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))
from keras_drag_predictor import build_drag_predictor
from keras_steady_flow_predictor import build_flow_unet


def test_drag_predictor_architecture():
    input_shape = (42, 222, 1)
    model = build_drag_predictor(input_shape)

    assert model is not None
    assert model.input_shape == (None, 42, 222, 1)
    assert model.output_shape == (None, 1)

    # Test forward pass with dummy tensor
    dummy_input = np.random.randn(2, 42, 222, 1).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, 1)
    assert not np.isnan(output.numpy()).any()


def test_flow_unet_architecture():
    input_shape = (42, 222, 1)
    model = build_flow_unet(input_shape)

    assert model is not None
    assert model.input_shape == (None, 42, 222, 1)
    assert model.output_shape == (None, 42, 222, 3)

    # Test forward pass with dummy tensor
    dummy_input = np.random.randn(2, 42, 222, 1).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, 42, 222, 3)
    assert not np.isnan(output.numpy()).any()
