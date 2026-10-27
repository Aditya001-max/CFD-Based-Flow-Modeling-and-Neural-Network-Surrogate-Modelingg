#!/usr/bin/env python3
"""
VTK/VTM CFD Simulation Data Processing Pipeline
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
Repository: https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples

Provides dataset loading, VTK multiblock parsing, coordinate rasterization,
and minibatch generation for CFD surrogate neural networks.
"""

import os
import glob
import logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VTKDataPipeline")


class VTKDataset:
    """
    Parser and data loader for OpenLB CFD outputs stored as VTK MultiBlock (.vtm)
    and Gnuplot drag force records.
    """

    def __init__(self, base_dir: str = "../data", train_test_split: float = 0.8):
        """
        Args:
            base_dir: Root directory containing 'xml_runs' and 'simulation_data' subdirectories.
            train_test_split: Fraction of samples allocated to training (default: 0.8).
        """
        self.base_dir = os.path.abspath(base_dir)
        self.train_test_split = train_test_split

        self.geometries: List[np.ndarray] = []
        self.steady_flows: List[np.ndarray] = []
        self.drag_vectors: List[np.ndarray] = []
        self.split_line: int = 0

    def load_data(self) -> int:
        """
        Parses all run configurations and extracts geometry, flow, and drag fields.
        Returns the number of successfully loaded samples.
        """
        xml_pattern = os.path.join(self.base_dir, "xml_runs", "*.xml")
        xml_files = sorted(glob.glob(xml_pattern))

        if not xml_files:
            logger.warning(f"No XML configuration files detected in {xml_pattern}.")
            return 0

        logger.info(f"Discovered {len(xml_files)} simulation manifests. Beginning parsing...")

        for f in xml_files:
            try:
                tree = ET.parse(f)
                root = tree.getroot()
                save_path_elem = root.find("save_path")
                if save_path_elem is None:
                    continue
                dirname_elem = save_path_elem.find("dirname")
                if dirname_elem is None or not dirname_elem.text:
                    continue
                dir_name = dirname_elem.text.strip()

                geometry_file = os.path.join(dir_name, "vtkData", "geometry_iT0000000.vtm")
                steady_flow_files = glob.glob(os.path.join(dir_name, "vtkData", "data", "*.vtm"))
                drag_vector_file = os.path.join(dir_name, "gnuplotData", "data", "drag.dat")

                if not (os.path.exists(geometry_file) and steady_flow_files and os.path.exists(drag_vector_file)):
                    continue

                steady_flow_file = steady_flow_files[0]

                # Parse geometry field
                geometry_array = self._read_vtm_geometry(geometry_file)
                if geometry_array is None or np.isnan(geometry_array).any():
                    continue

                # Parse steady state flow field (velocity [u, v] + pressure [p])
                steady_flow_array = self._read_vtm_flow(steady_flow_file)
                if steady_flow_array is None or np.isnan(steady_flow_array).any():
                    continue

                # Parse drag vector
                drag_array = self._read_drag_dat(drag_vector_file)
                if drag_array is None or np.isnan(drag_array).any():
                    continue

                self.geometries.append(geometry_array)
                self.steady_flows.append(steady_flow_array)
                self.drag_vectors.append(drag_array)

            except Exception as e:
                logger.debug(f"Skipping record due to parse exception: {e}")
                continue

        self.split_line = int(self.train_test_split * len(self.geometries))
        logger.info(
            f"Successfully parsed {len(self.geometries)} samples "
            f"({self.split_line} train, {len(self.geometries) - self.split_line} test)."
        )
        return len(self.geometries)

    @staticmethod
    def _read_vtm_geometry(filepath: str) -> Optional[np.ndarray]:
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required to parse .vtm geometry files.")
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        data = reader.GetOutput()
        data_iterator = data.NewIterator()
        img_data = data_iterator.GetCurrentDataObject()
        if hasattr(img_data, "Update"):
            img_data.Update()

        point_data = img_data.GetPointData()
        array_data = point_data.GetArray(0)
        np_array = vtk_to_numpy(array_data)
        img_shape = img_data.GetWholeExtent()
        target_shape = [img_shape[3] - img_shape[2] + 1, img_shape[1] - img_shape[0] + 1, 1]
        return np_array.reshape(target_shape)

    @staticmethod
    def _read_vtm_flow(filepath: str) -> Optional[np.ndarray]:
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required to parse .vtm flow files.")
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        data = reader.GetOutput()
        data_iterator = data.NewIterator()
        img_data = data_iterator.GetCurrentDataObject()
        if hasattr(img_data, "Update"):
            img_data.Update()

        point_data = img_data.GetPointData()
        vel_arr = vtk_to_numpy(point_data.GetArray(0))
        press_arr = vtk_to_numpy(point_data.GetArray(1))

        img_shape = img_data.GetWholeExtent()
        h = img_shape[3] - img_shape[2] + 1
        w = img_shape[1] - img_shape[0] + 1

        vel_np = vel_arr.reshape([h, w, 2])
        press_np = press_arr.reshape([h, w, 1])
        return np.concatenate([vel_np, press_np], axis=2)

    @staticmethod
    def _read_drag_dat(filepath: str) -> Optional[np.ndarray]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return None
        drag_values = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    drag_values.append(float(parts[1]))
                except ValueError:
                    continue
        if not drag_values:
            return None
        return np.array(drag_values, dtype=np.float32)

    def generate_synthetic_data(
        self,
        num_samples: int = 100,
        height: int = 42,
        width: int = 222
    ) -> None:
        """
        Synthesizes realistic CFD flow and drag profiles around cylindrical obstacles.
        Enables testing, benchmark validation, and pipeline verification without external solvers.
        """
        logger.info(f"Synthesizing {num_samples} realistic flow samples [{height}x{width}].")
        self.geometries = []
        self.steady_flows = []
        self.drag_vectors = []

        y_coords, x_coords = np.mgrid[0:height, 0:width]

        for i in range(num_samples):
            # Geometry: 1 for fluid, 0 for solid cylinder obstacle
            geom = np.ones((height, width, 1), dtype=np.float32)
            radius = np.random.uniform(height * 0.12, height * 0.28)
            cx = np.random.uniform(width * 0.15, width * 0.45)
            cy = np.random.uniform(radius + 2, height - radius - 2)

            dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            geom[dist <= radius, 0] = 0.0

            # Poiseuille background flow: u_inlet(y) = 4 * U_max * y * (H - y) / H^2
            norm_y = y_coords / float(height - 1)
            u_base = 4.0 * 1.5 * norm_y * (1.0 - norm_y)

            # Potential flow / wake perturbation
            wake_factor = np.exp(-((x_coords - cx) / (width * 0.25)) ** 2) * (dist > radius)
            u_field = u_base * geom[:, :, 0] + 0.1 * np.sin(x_coords / 10.0) * wake_factor
            v_field = -0.05 * np.sin((y_coords - cy) / 5.0) * wake_factor * geom[:, :, 0]

            # Pressure field: High upstream of cylinder, low downstream
            p_field = (1.0 - x_coords / float(width)) * 2.0
            p_field[dist <= radius] = 0.0
            p_field[dist <= radius * 1.5] += 0.5

            flow_field = np.stack([u_field, v_field, p_field], axis=-1).astype(np.float32)

            # Steady drag coefficient proportional to diameter & Reynolds number dynamics
            base_drag = float(5.0 + 35.0 * (radius / float(height)) + np.random.normal(0, 0.2))
            drag_history = np.linspace(base_drag * 0.8, base_drag, 50).astype(np.float32)

            self.geometries.append(geom)
            self.steady_flows.append(flow_field)
            self.drag_vectors.append(drag_history)

        self.split_line = int(self.train_test_split * len(self.geometries))
        logger.info(f"Synthetic generation complete: {len(self.geometries)} samples generated.")

    def minibatch(
        self,
        train: bool = True,
        batch_size: int = 32,
        batch_type: str = "flow"
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Samples a randomized minibatch for training or validation.
        """
        if not self.geometries:
            raise ValueError("No data available. Call load_data() or generate_synthetic_data() first.")

        batch_boundary = []
        batch_data = []

        for _ in range(batch_size):
            if train:
                sample_idx = np.random.randint(0, max(1, self.split_line))
            else:
                sample_idx = np.random.randint(self.split_line, len(self.geometries))

            batch_boundary.append(self.geometries[sample_idx])
            if batch_type == "flow":
                batch_data.append(self.steady_flows[sample_idx])
            elif batch_type == "drag":
                batch_data.append(self.drag_vectors[sample_idx])
            else:
                raise ValueError(f"Unknown batch_type: {batch_type}. Choose 'flow' or 'drag'.")

        return batch_boundary, batch_data


# Backward-compatible alias
VTK_data = VTKDataset
