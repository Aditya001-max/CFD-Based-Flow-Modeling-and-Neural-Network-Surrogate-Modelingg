"""
Unit Tests for VTK / VTM Data Loader & Synthetic CFD Generator
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
"""

import sys
import os
import pytest
import numpy as np

# Ensure train directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))
from vtm_data import VTKDataset, VTK_data


def test_synthetic_data_generation():
    dataset = VTKDataset(train_test_split=0.75)
    num_samples = 20
    h, w = 42, 222
    dataset.generate_synthetic_data(num_samples=num_samples, height=h, width=w)

    assert len(dataset.geometries) == num_samples
    assert len(dataset.steady_flows) == num_samples
    assert len(dataset.drag_vectors) == num_samples
    assert dataset.split_line == int(0.75 * num_samples)

    # Validate shapes
    assert dataset.geometries[0].shape == (h, w, 1)
    assert dataset.steady_flows[0].shape == (h, w, 3)
    assert dataset.drag_vectors[0].ndim == 1

    # Validate no NaNs
    assert not np.isnan(dataset.geometries[0]).any()
    assert not np.isnan(dataset.steady_flows[0]).any()
    assert not np.isnan(dataset.drag_vectors[0]).any()


def test_minibatch_flow_and_drag():
    dataset = VTKDataset()
    dataset.generate_synthetic_data(num_samples=16)

    # Train batch for flow
    b_geom, b_flow = dataset.minibatch(train=True, batch_size=4, batch_type="flow")
    assert len(b_geom) == 4
    assert len(b_flow) == 4
    assert b_geom[0].shape == (42, 222, 1)
    assert b_flow[0].shape == (42, 222, 3)

    # Test batch for drag
    b_geom_test, b_drag = dataset.minibatch(train=False, batch_size=4, batch_type="drag")
    assert len(b_geom_test) == 4
    assert len(b_drag) == 4

    # Backward compatibility alias
    assert VTK_data is VTKDataset


def test_invalid_batch_type_raises():
    dataset = VTKDataset()
    dataset.generate_synthetic_data(num_samples=5)
    with pytest.raises(ValueError):
        dataset.minibatch(batch_type="invalid_type")
