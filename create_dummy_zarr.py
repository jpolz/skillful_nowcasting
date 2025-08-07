#!/usr/bin/env python3
"""
Generate a dummy Zarr dataset for testing DGMR training pipeline.

This script creates synthetic precipitation radar data in Zarr format
that can be used to test the training system without real data.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


def generate_synthetic_precipitation(
    time_steps: int,
    height: int,
    width: int,
    spatial_correlation: float = 0.8,
    temporal_correlation: float = 0.9,
    max_intensity: float = 50.0,
    storm_probability: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic precipitation data with realistic spatial and temporal patterns.

    Args:
        time_steps: Number of time steps
        height: Spatial height dimension
        width: Spatial width dimension
        spatial_correlation: Spatial correlation strength (0-1)
        temporal_correlation: Temporal correlation strength (0-1)
        max_intensity: Maximum precipitation intensity (mm/hour)
        storm_probability: Probability of storm cells appearing
        seed: Random seed for reproducibility

    Returns:
        Precipitation data array of shape (time, height, width)
    """
    np.random.seed(seed)

    # Initialize array
    precip = np.zeros((time_steps, height, width), dtype=np.float32)

    # Generate base noise field
    noise = np.random.randn(time_steps, height, width)

    # Apply spatial smoothing for realistic precipitation patterns
    from scipy.ndimage import gaussian_filter

    for t in range(time_steps):
        # Smooth spatially for realistic storm structures
        smooth_noise = gaussian_filter(noise[t], sigma=spatial_correlation * 5)

        # Apply temporal correlation
        if t > 0:
            # Blend with previous time step
            smooth_noise = (
                temporal_correlation * precip[t - 1]
                + (1 - temporal_correlation) * smooth_noise
            )

        # Create storm cells
        storm_mask = np.random.random((height, width)) < storm_probability
        storm_intensity = np.random.exponential(
            scale=max_intensity / 3, size=(height, width)
        )

        # Combine smooth background with storm cells
        base_field = np.maximum(0, smooth_noise)
        storm_field = storm_mask * storm_intensity

        # Final precipitation field
        precip[t] = np.maximum(base_field, storm_field)

        # Apply realistic precipitation distribution (mostly light, some heavy)
        precip[t] = np.where(precip[t] > 0, np.random.gamma(2, precip[t] / 2), 0)

        # Cap maximum intensity
        precip[t] = np.clip(precip[t], 0, max_intensity)

    return precip


def create_dummy_zarr_dataset(
    output_path: str,
    time_steps: int = 500,
    height: int = 256,
    width: int = 256,
    time_interval_minutes: int = 5,
    chunk_time: int = 50,
    chunk_spatial: int = 128,
    seed: int = 42,
) -> None:
    """
    Create a dummy Zarr dataset for testing DGMR training.

    Args:
        output_path: Path where to save the Zarr dataset
        time_steps: Number of time steps to generate
        height: Spatial height dimension
        width: Spatial width dimension
        time_interval_minutes: Time interval between frames in minutes
        chunk_time: Chunk size for time dimension
        chunk_spatial: Chunk size for spatial dimensions
        seed: Random seed for reproducibility
    """
    print(f"Generating dummy Zarr dataset: {output_path}")
    print(f"Shape: {time_steps} × {height} × {width}")
    print(f"Time interval: {time_interval_minutes} minutes")

    # Generate time coordinates
    start_time = datetime(2023, 6, 1, 0, 0, 0)
    time_coords = [
        start_time + timedelta(minutes=i * time_interval_minutes)
        for i in range(time_steps)
    ]

    # Generate spatial coordinates (in km, covering ~1280km x 1280km area)
    x_coords = np.linspace(0, 1280, width)
    y_coords = np.linspace(0, 1280, height)

    print("Generating synthetic precipitation data...")

    # Generate synthetic precipitation data
    try:
        precipitation = generate_synthetic_precipitation(
            time_steps=time_steps, height=height, width=width, seed=seed
        )
    except ImportError:
        # Fallback if scipy is not available
        print("Warning: scipy not available, using simpler generation method")
        np.random.seed(seed)

        # Simple moving average approach for spatial/temporal correlation
        precipitation = np.zeros((time_steps, height, width), dtype=np.float32)

        for t in range(time_steps):
            # Generate base random field
            base = np.random.exponential(scale=2.0, size=(height, width))

            # Add some spatial structure with simple smoothing
            if height > 3 and width > 3:
                # Simple 3x3 averaging for spatial correlation
                padded = np.pad(base, 1, mode="edge")
                for i in range(1, height + 1):
                    for j in range(1, width + 1):
                        base[i - 1, j - 1] = np.mean(
                            padded[i - 1 : i + 2, j - 1 : j + 2]
                        )

            # Add temporal correlation
            if t > 0:
                base = 0.7 * precipitation[t - 1] + 0.3 * base

            # Apply precipitation-like distribution
            storm_prob = 0.2
            storm_mask = np.random.random((height, width)) < storm_prob
            precipitation[t] = np.where(storm_mask, base * 5, base * 0.5)
            precipitation[t] = np.clip(precipitation[t], 0, 50)

    print(
        f"Generated precipitation range: {precipitation.min():.2f} - {precipitation.max():.2f} mm/hour"
    )
    print(f"Mean precipitation: {precipitation.mean():.2f} mm/hour")

    # Create xarray Dataset
    print("Creating xarray Dataset...")
    dataset = xr.Dataset(
        {
            "precipitation": (
                ["time", "y", "x"],
                precipitation,
                {
                    "units": "mm/hour",
                    "long_name": "Precipitation rate",
                    "standard_name": "precipitation_flux",
                },
            )
        },
        coords={
            "time": ("time", time_coords),
            "x": ("x", x_coords, {"units": "km", "long_name": "Easting"}),
            "y": ("y", y_coords, {"units": "km", "long_name": "Northing"}),
        },
        attrs={
            "title": "Dummy precipitation dataset for DGMR testing",
            "source": "Synthetically generated",
            "created": datetime.now().isoformat(),
            "conventions": "CF-1.8",
        },
    )

    # Configure chunking for efficient access
    chunks = {"time": chunk_time, "y": chunk_spatial, "x": chunk_spatial}

    chunked_dataset = dataset.chunk(chunks)

    print(f"Saving to Zarr with chunking: {chunks}")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing dataset if it exists
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    # Save to Zarr
    encoding = {"precipitation": {"dtype": "float32"}}

    chunked_dataset.to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)

    print("✅ Dummy Zarr dataset created successfully!")
    print(f"Location: {output_path.absolute()}")

    # Print dataset info
    print("\nDataset Summary:")
    print(f"  Time steps: {len(dataset.time)}")
    print(f"  Spatial resolution: {height} × {width}")
    print(f"  Data variables: {list(dataset.data_vars.keys())}")
    print(f"  File size: {get_directory_size(output_path):.1f} MB")

    # Test loading a sample
    print("\nTesting dataset loading...")
    test_load = xr.open_zarr(output_path)
    sample = test_load.precipitation.isel(time=0).values
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample range: {sample.min():.2f} - {sample.max():.2f}")
    print("  ✅ Dataset loads successfully!")


def get_directory_size(path: Path) -> float:
    """Get the total size of a directory in MB."""
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)


def validate_dataset_for_training(zarr_path: str, min_frames: int = 22) -> bool:
    """
    Validate that the dataset can be used for DGMR training.

    Args:
        zarr_path: Path to the Zarr dataset
        min_frames: Minimum frames needed (input + target)

    Returns:
        True if dataset is valid for training
    """
    print(f"\nValidating dataset for training: {zarr_path}")

    try:
        dataset = xr.open_zarr(zarr_path)

        # Check required dimensions
        required_dims = ["time"]
        for dim in required_dims:
            if dim not in dataset.dims:
                print(f"❌ Missing required dimension: {dim}")
                return False

        # Check minimum time steps
        time_steps = len(dataset.time)
        if time_steps < min_frames:
            print(f"❌ Insufficient time steps: {time_steps} < {min_frames}")
            return False

        # Check for data variables
        if len(dataset.data_vars) == 0:
            print("❌ No data variables found")
            return False

        # Check spatial dimensions
        data_var = list(dataset.data_vars.keys())[0]
        data_shape = dataset[data_var].shape

        if len(data_shape) < 3:
            print(f"❌ Data must have at least 3 dimensions, got {len(data_shape)}")
            return False

        print("✅ Dataset validation passed!")
        print(f"  Time steps: {time_steps}")
        print(f"  Data shape: {data_shape}")
        print(f"  Data variables: {list(dataset.data_vars.keys())}")
        print(f"  Spatial dimensions: {data_shape[1:]} (after time)")

        return True

    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False


def main():
    """Main function for generating dummy Zarr datasets."""
    parser = argparse.ArgumentParser(
        description="Generate dummy Zarr dataset for DGMR training tests"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/dummy_radar_dataset.zarr",
        help="Output path for the Zarr dataset",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=50,
        help="Number of time steps to generate (default: 500)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Spatial height dimension (default: 256)",
    )
    parser.add_argument(
        "--width", type=int, default=256, help="Spatial width dimension (default: 256)"
    )
    parser.add_argument(
        "--time_interval",
        type=int,
        default=5,
        help="Time interval between frames in minutes (default: 5)",
    )
    parser.add_argument(
        "--chunk_time",
        type=int,
        default=500,
        help="Chunk size for time dimension (default: 50)",
    )
    parser.add_argument(
        "--chunk_spatial",
        type=int,
        default=128,
        help="Chunk size for spatial dimensions (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the dataset after creation"
    )

    args = parser.parse_args()

    # Generate the dataset
    create_dummy_zarr_dataset(
        output_path=args.output,
        time_steps=args.time_steps,
        height=args.height,
        width=args.width,
        time_interval_minutes=args.time_interval,
        chunk_time=args.chunk_time,
        chunk_spatial=args.chunk_spatial,
        seed=args.seed,
    )

    # Validate if requested
    if args.validate:
        validate_dataset_for_training(args.output)

    print("\n🎉 Dummy dataset ready for testing!")
    print("Usage example:")
    print(f"  python train/run.py --config config/debug.yml --zarr_path {args.output}")


if __name__ == "__main__":
    main()
