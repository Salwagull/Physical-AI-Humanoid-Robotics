#!/usr/bin/env python3
"""
Isaac Sim Synthetic Data Generation for Perception Training

This script demonstrates how to use Omniverse Replicator to generate training
data for object detection, semantic segmentation, and depth estimation models.

Features:
- Domain randomization (lighting, textures, poses, camera)
- Multiple output formats (RGB, depth, segmentation, bounding boxes)
- COCO-format annotation export
- Configurable number of samples

Prerequisites:
- NVIDIA Isaac Sim 4.5+ installed
- RTX GPU with 8GB+ VRAM

Usage:
    # Generate 1000 images
    python synthetic_data_generation.py --num_frames 1000

    # Generate with specific output directory
    python synthetic_data_generation.py --output_dir ./my_dataset

    # Generate headless (faster)
    python synthetic_data_generation.py --headless --num_frames 10000

Output Structure:
    ./synthetic_data/
    ├── rgb/                    # RGB images
    │   ├── 000000.png
    │   └── ...
    ├── depth/                  # Depth images (meters)
    │   ├── 000000.png
    │   └── ...
    ├── semantic_segmentation/  # Semantic masks
    │   ├── 000000.png
    │   └── ...
    ├── instance_segmentation/  # Instance masks
    │   ├── 000000.png
    │   └── ...
    ├── bounding_box_2d_tight/  # Bounding box annotations
    │   ├── 000000.json
    │   └── ...
    └── annotations.json        # COCO-format annotations

Author: Physical AI & Humanoid Robotics Book
"""

import argparse
import os
import json
from datetime import datetime

# Parse arguments before initializing simulation
parser = argparse.ArgumentParser(description="Generate synthetic training data")
parser.add_argument("--num_frames", type=int, default=1000, help="Number of images to generate")
parser.add_argument("--output_dir", type=str, default="./synthetic_data", help="Output directory")
parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720], help="Image resolution [width height]")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# Initialize Isaac Sim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "RayTracedLighting",  # Enable RTX for realistic rendering
    "width": args.resolution[0],
    "height": args.resolution[1],
})

print("="*60)
print("Synthetic Data Generation with Isaac Sim")
print("="*60)
print(f"Number of frames: {args.num_frames}")
print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
print(f"Output directory: {args.output_dir}")
print(f"Headless: {args.headless}")
print("="*60)

# Import Omniverse modules
import omni
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.isaac.core.utils.prims as prim_utils
import numpy as np

# Set random seed for reproducibility
np.random.seed(args.seed)


def setup_scene():
    """Create the simulation scene with objects to detect."""

    print("\nSetting up scene...")

    # Create world
    world = World()
    world.scene.add_default_ground_plane()

    # Get assets root path
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("Warning: Could not find Isaac Sim assets. Using basic primitives.")
        assets_root = ""

    # Add a table
    table = rep.create.cube(
        position=(0, 0, 0.4),
        scale=(1.2, 0.8, 0.05),
        semantics=[("class", "table")],
        name="Table"
    )
    print("  ✓ Table added")

    # YCB objects to spawn on the table
    ycb_objects = [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "037_scissors",
    ]

    # Create object pool
    objects = []
    for i, obj_name in enumerate(ycb_objects[:5]):  # Use first 5 objects
        usd_path = f"{assets_root}/Isaac/Props/YCB/{obj_name}.usd"

        try:
            obj = rep.create.from_usd(
                usd_files=[usd_path],
                semantics=[("class", obj_name.split("_", 1)[1])],
                count=1,
            )
            objects.append(obj)
            print(f"  ✓ Added {obj_name}")
        except Exception as e:
            # Fall back to primitive shapes if USD not found
            print(f"  ⚠ Could not load {obj_name}, using primitive instead")
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
            obj = rep.create.cube(
                position=(0, 0, 0.5),
                scale=(0.05 + i*0.01, 0.05 + i*0.01, 0.1),
                semantics=[("class", f"object_{i}")],
                material=rep.create.material_omnipbr(diffuse=colors[i % len(colors)])
            )
            objects.append(obj)

    return world, table, objects


def setup_camera():
    """Create and configure the camera for data capture."""

    print("\nSetting up camera...")

    # Create camera with initial position
    camera = rep.create.camera(
        position=(2.0, 2.0, 1.5),
        look_at=(0, 0, 0.5),
        focal_length=35.0,
    )

    # Create render product (what the camera sees)
    render_product = rep.create.render_product(
        camera,
        resolution=tuple(args.resolution)
    )

    print(f"  ✓ Camera created at initial position")
    print(f"  ✓ Render product: {args.resolution[0]}x{args.resolution[1]}")

    return camera, render_product


def setup_lighting():
    """Create lights for the scene."""

    print("\nSetting up lighting...")

    # Dome light for ambient illumination
    dome_light = rep.create.light(
        light_type="dome",
        intensity=1000,
        temperature=6500,  # Daylight color temperature
    )

    # Directional light for shadows
    directional_light = rep.create.light(
        light_type="distant",
        intensity=3000,
        temperature=5500,
        rotation=(45, 45, 0),
    )

    print("  ✓ Dome light added")
    print("  ✓ Directional light added")

    return dome_light, directional_light


def setup_randomization(camera, objects, lights):
    """Configure domain randomization triggers."""

    print("\nSetting up domain randomization...")

    dome_light, directional_light = lights

    with rep.trigger.on_frame():
        # Randomize object positions on table
        for obj in objects:
            with obj:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-0.4, -0.3, 0.45),
                        (0.4, 0.3, 0.55)
                    ),
                    rotation=rep.distribution.uniform(
                        (0, 0, 0),
                        (0, 0, 360)
                    ),
                    scale=rep.distribution.uniform(
                        (0.8, 0.8, 0.8),
                        (1.2, 1.2, 1.2)
                    ),
                )

        # Randomize camera position (orbit around table)
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (1.5, 1.5, 1.0),
                    (3.0, 3.0, 2.5)
                ),
                look_at=(0, 0, 0.4),
            )

        # Randomize lighting
        with dome_light:
            rep.modify.attribute("intensity", rep.distribution.uniform(500, 2000))
            rep.modify.attribute("color", rep.distribution.uniform((0.9, 0.9, 0.9), (1.1, 1.1, 1.0)))

        with directional_light:
            rep.modify.attribute("intensity", rep.distribution.uniform(1500, 5000))
            rep.modify.pose(rotation=rep.distribution.uniform((30, 0, 0), (60, 360, 0)))

    print("  ✓ Object pose randomization")
    print("  ✓ Camera position randomization")
    print("  ✓ Lighting randomization")


def setup_writers(render_product, output_dir):
    """Configure output writers for different data modalities."""

    print("\nSetting up data writers...")

    os.makedirs(output_dir, exist_ok=True)

    # Basic writer for RGB, depth, and segmentation
    basic_writer = rep.WriterRegistry.get("BasicWriter")
    basic_writer.initialize(
        output_dir=output_dir,
        rgb=True,
        distance_to_image_plane=True,  # Depth
        semantic_segmentation=True,
        instance_segmentation=True,
        bounding_box_2d_tight=True,
        colorize_semantic_segmentation=True,
        colorize_instance_segmentation=True,
    )
    basic_writer.attach([render_product])

    print("  ✓ RGB images")
    print("  ✓ Depth images")
    print("  ✓ Semantic segmentation")
    print("  ✓ Instance segmentation")
    print("  ✓ 2D bounding boxes")

    return basic_writer


def generate_data(num_frames):
    """Main data generation loop."""

    print("\n" + "="*60)
    print("STARTING DATA GENERATION")
    print("="*60)

    # Progress tracking
    start_time = datetime.now()
    log_interval = max(1, num_frames // 20)  # Log 20 times

    for i in range(num_frames):
        # Step the replicator (applies randomization and captures)
        rep.orchestrator.step()

        # Progress logging
        if (i + 1) % log_interval == 0 or i == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (num_frames - i - 1) / fps if fps > 0 else 0
            print(f"  Generated {i+1}/{num_frames} frames "
                  f"({100*(i+1)/num_frames:.1f}%) - "
                  f"{fps:.1f} FPS - "
                  f"ETA: {remaining:.0f}s")

    # Final stats
    total_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {num_frames/total_time:.1f}")
    print(f"Output directory: {args.output_dir}")


def create_coco_annotations(output_dir):
    """Convert bounding box annotations to COCO format."""

    print("\nCreating COCO annotations...")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build category list from semantic classes
    category_map = {}
    category_id = 1

    bbox_dir = os.path.join(output_dir, "bounding_box_2d_tight")
    if not os.path.exists(bbox_dir):
        print("  ⚠ No bounding box directory found, skipping COCO export")
        return

    annotation_id = 1

    for filename in sorted(os.listdir(bbox_dir)):
        if not filename.endswith(".json"):
            continue

        image_id = int(filename.split(".")[0])

        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "file_name": f"rgb/{image_id:06d}.png",
            "width": args.resolution[0],
            "height": args.resolution[1],
        })

        # Parse bounding boxes
        with open(os.path.join(bbox_dir, filename)) as f:
            bbox_data = json.load(f)

        for bbox in bbox_data.get("data", []):
            class_name = bbox.get("semanticLabel", "unknown")

            # Add category if new
            if class_name not in category_map:
                category_map[class_name] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": "object"
                })
                category_id += 1

            # Convert to COCO format [x, y, width, height]
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]
            width = x_max - x_min
            height = y_max - y_min

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[class_name],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

    # Save COCO annotations
    coco_path = os.path.join(output_dir, "annotations_coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"  ✓ COCO annotations saved to: {coco_path}")
    print(f"  ✓ {len(coco_data['images'])} images")
    print(f"  ✓ {len(coco_data['annotations'])} annotations")
    print(f"  ✓ {len(coco_data['categories'])} categories")


def main():
    """Main function."""

    try:
        # Setup scene
        world, table, objects = setup_scene()

        # Setup camera
        camera, render_product = setup_camera()

        # Setup lighting
        lights = setup_lighting()

        # Setup randomization
        setup_randomization(camera, objects, lights)

        # Setup writers
        writer = setup_writers(render_product, args.output_dir)

        # Reset world
        world.reset()

        # Generate data
        generate_data(args.num_frames)

        # Create COCO annotations
        create_coco_annotations(args.output_dir)

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")

    finally:
        simulation_app.close()

    print("\n" + "="*60)
    print("To train an object detection model with this data:")
    print("  1. Install a detection framework (e.g., detectron2, YOLOv8)")
    print("  2. Point it to the COCO annotations file")
    print("  3. Train on the synthetic data")
    print("  4. Fine-tune on real data for best results")
    print("="*60)


if __name__ == "__main__":
    main()
