"""
Unit Tests for CFD Dataset Generator & XML Serialization
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
"""

import sys
import os
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from make_dataset import generate_dataset


def test_xml_manifest_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        xml_dir = os.path.join(tmp_dir, "xml_runs")
        data_dir = os.path.join(tmp_dir, "simulation_data")

        num_runs = 5
        generate_dataset(
            num_runs=num_runs,
            num_queue=2,
            xml_dir=xml_dir,
            data_dir=data_dir,
            binary_path="./cylinder2d/cylinder2d",
            dry_run=True
        )

        generated_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
        assert len(generated_files) == num_runs

        # Inspect first XML file
        first_xml = os.path.join(xml_dir, generated_files[0])
        tree = ET.parse(first_xml)
        root = tree.getroot()

        mesh = root.find("Mesh")
        assert mesh is not None
        assert float(mesh.find("x_pos").text) >= 0.20
        assert float(mesh.find("y_pos").text) > 0.0
        assert float(mesh.find("radius").text) >= 0.05

        save_path = root.find("save_path")
        assert save_path is not None
        assert "runlog_" in save_path.find("dirname").text
