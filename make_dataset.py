#!/usr/bin/env python3
"""
CFD Dataset Generation Pipeline
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
Repository: https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples

Generates 2D cylinder CFD simulation parameter definitions (XML), orchestrates OpenLB
lattice Boltzmann solver execution across concurrent worker threads, and structures raw
simulation outputs (VTK/VTM and drag force data) for supervised deep learning surrogate models.
"""

import os
import sys
import argparse
import subprocess
import logging
import xml.etree.cElementTree as ET
import numpy as np

try:
    from queue import Queue
except ImportError:
    from Queue import Queue
from threading import Thread

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CFDDatasetGenerator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFD Dataset Generation Orchestrator (OpenLB Lattice Boltzmann)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2000,
        help="Total number of simulation geometry configurations to generate (default: 2000)"
    )
    parser.add_argument(
        "--num-queue",
        type=int,
        default=8,
        help="Number of concurrent worker queue threads for simulations (default: 8)"
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default="./data/xml_runs",
        help="Directory to store XML simulation parameter configurations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/simulation_data",
        help="Directory to store simulation run output data (VTK & drag dat files)"
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="./cylinder2d/cylinder2d",
        help="Path to compiled OpenLB cylinder2d simulation executable"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate XML parameter files without launching external C++ binaries"
    )
    return parser.parse_args()


def worker(simulation_queue):
    """Worker thread that consumes simulation commands from the queue."""
    while True:
        item = simulation_queue.get()
        if item is None:
            simulation_queue.task_done()
            break
        sim_cmd, rm_cmd = item
        with open(os.devnull, "w") as devnull:
            try:
                # Remove previous corrupted/partial simulation run directory if exists
                if os.name == "nt":
                    # Windows cleanup fallback
                    dirname = rm_cmd.replace("rm -r ", "").strip()
                    if os.path.exists(dirname):
                        import shutil
                        shutil.rmtree(dirname, ignore_errors=True)
                else:
                    subprocess.check_call(rm_cmd.split(), stdout=devnull, stderr=devnull)
            except Exception as e:
                logger.debug(f"Pre-cleanup notice: {e}")

            try:
                subprocess.check_call(sim_cmd.split(), stdout=devnull, stderr=devnull)
            except Exception as err:
                logger.error(f"Simulation execution error: {err} | Command: {sim_cmd}")
            finally:
                simulation_queue.task_done()


def generate_dataset(num_runs, num_queue, xml_dir, data_dir, binary_path, dry_run=False):
    """Creates geometric configurations, XML parameter files, and schedules simulation tasks."""
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    abs_data_dir = os.path.abspath(data_dir)
    logger.info(f"Initiating CFD dataset generation: {num_runs} runs with {num_queue} workers.")

    simulation_queue = Queue(maxsize=num_queue * 2)

    # Spawn worker threads if not dry-run
    threads = []
    if not dry_run:
        for _ in range(num_queue):
            t = Thread(target=worker, args=(simulation_queue,))
            t.daemon = True
            t.start()
            threads.append(t)

    for i in tqdm(range(num_runs), desc="Generating CFD Cases"):
        # Stochastic cylinder placement and diameter within 2D flow channel [0, 2.2] x [0, 0.41]
        rand_x = float(np.random.uniform(0.20, 1.0))
        rand_radius = float(np.random.uniform(0.05, 0.15))
        rand_y = float(np.random.uniform(0.0 + rand_radius + 0.01, 0.41 - rand_radius - 0.01))

        run_id = str(i).zfill(5)
        run_output_dir = os.path.join(abs_data_dir, f"runlog_{run_id}") + "/"
        xml_file_path = os.path.join(xml_dir, f"run_{run_id}.xml")

        # Construct XML parameter specification for OpenLB simulation
        root = ET.Element("Param")
        mesh_elem = ET.SubElement(root, "Mesh")
        ET.SubElement(mesh_elem, "x_pos").text = f"{rand_x:.6f}"
        ET.SubElement(mesh_elem, "y_pos").text = f"{rand_y:.6f}"
        ET.SubElement(mesh_elem, "radius").text = f"{rand_radius:.6f}"

        save_path_elem = ET.SubElement(root, "save_path")
        ET.SubElement(save_path_elem, "dirname").text = run_output_dir

        tree = ET.ElementTree(root)
        tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)

        if not dry_run:
            sim_cmd = f"{binary_path} {xml_file_path}"
            rm_cmd = f"rm -r {run_output_dir.rstrip('/')}"
            simulation_queue.put([sim_cmd, rm_cmd])

    if not dry_run:
        simulation_queue.join()
        logger.info("All CFD simulations successfully completed.")
    else:
        logger.info(f"Dry-run complete: {num_runs} XML configuration files generated in {xml_dir}.")


def main():
    args = parse_args()
    generate_dataset(
        num_runs=args.num_runs,
        num_queue=args.num_queue,
        xml_dir=args.xml_dir,
        data_dir=args.data_dir,
        binary_path=args.binary,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
