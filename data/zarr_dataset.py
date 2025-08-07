"""Simple Zarr dataset for DGMR training."""

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class ZarrNowcastingDataset(Dataset):
    """Simple Zarr dataset for nowcasting with DGMR."""

    def __init__(
        self,
        zarr_path: str,
        input_frames: int = 4,
        target_frames: int = 18,
        transform=None,
        indices=None,
    ):
        """
        Initialize Zarr dataset.

        Args:
            zarr_path: Path to Zarr store
            input_frames: Number of input frames
            target_frames: Number of target frames
            transform: Optional transform function
            indices: Optional list of time indices to use
        """
        self.zarr_path = zarr_path
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.transform = transform

        # Open zarr dataset
        self.dataset = xr.open_zarr(zarr_path, chunks="auto")
        self.total_time_steps = len(self.dataset.time)

        # Calculate valid starting indices (need input_frames + target_frames consecutive frames)
        total_frames_needed = input_frames + target_frames
        max_start_idx = self.total_time_steps - total_frames_needed

        if max_start_idx < 0:
            raise ValueError(
                f"Dataset has {self.total_time_steps} time steps, but need at least "
                f"{total_frames_needed} for {input_frames} input + {target_frames} target frames"
            )

        # Create all possible valid starting indices
        all_valid_indices = list(range(max_start_idx + 1))

        # Filter by provided indices if given
        if indices is not None:
            # Only keep indices that allow for complete sequences
            self.valid_indices = [idx for idx in indices if idx <= max_start_idx]
        else:
            self.valid_indices = all_valid_indices

        print(
            f"Dataset initialized: {len(self.valid_indices)} valid samples from {self.total_time_steps} time steps"
        )

    def __len__(self):
        """Return number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a sequence of input and target frames.

        Returns:
            tuple: (input_frames, target_frames) as numpy arrays
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.input_frames + self.target_frames

        # Extract sequence from the full dataset
        sequence = self.dataset.isel(time=slice(start_idx, end_idx))

        # Get the data variables from the dataset
        data_vars = list(sequence.data_vars.keys())
        print(f"Data variables in sequence: {data_vars}")

        if len(data_vars) == 1:
            # Single data variable - get its values
            var_name = data_vars[0]
            data = sequence[var_name].values.astype(np.float32)
        else:
            # Multiple data variables - stack along channel dimension
            data = np.stack(
                [sequence[var].values for var in data_vars], axis=-1
            ).astype(np.float32)

        # Ensure we have channel dimension: (time, height, width, channels)
        if data.ndim == 3:  # (time, height, width)
            data = data[..., np.newaxis]  # Add channel dimension

        # Split into input and target frames
        input_data = data[: self.input_frames]  # (input_frames, H, W, C)
        target_data = data[self.input_frames :]  # (target_frames, H, W, C)

        # Convert to PyTorch format: (frames, channels, height, width)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        target_data = np.transpose(target_data, (0, 3, 1, 2))

        # Apply transform if provided
        if self.transform:
            input_data = self.transform(input_data)
            target_data = self.transform(target_data)

        return torch.from_numpy(input_data), torch.from_numpy(target_data)


def create_train_val_split(zarr_path: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Create train/validation split indices for Zarr dataset.

    Args:
        zarr_path: Path to Zarr store
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducible splits

    Returns:
        tuple: (train_indices, val_indices)
    """
    # Open dataset to get time dimension
    ds = xr.open_zarr(zarr_path, chunks="auto")
    total_time_steps = len(ds.time)

    # Calculate valid starting indices (need input_frames + target_frames consecutive frames)
    # Assuming standard 4 input + 18 target = 22 frames needed
    frames_needed = 22  # This should match your model requirements
    max_start_idx = total_time_steps - frames_needed

    if max_start_idx < 0:
        raise ValueError(
            f"Dataset has {total_time_steps} time steps, but need at least "
            f"{frames_needed} consecutive frames for sequences"
        )

    # Create valid starting indices (these are the indices that can start a complete sequence)
    valid_start_indices = np.arange(max_start_idx + 1)

    # Shuffle and split the valid starting indices
    np.random.seed(seed)
    np.random.shuffle(valid_start_indices)

    # Split indices
    split_point = int(len(valid_start_indices) * train_ratio)
    train_indices = valid_start_indices[:split_point]
    val_indices = valid_start_indices[split_point:]

    return train_indices.tolist(), val_indices.tolist()


if __name__ == "__main__":
    # Test the dataset
    dataset = ZarrNowcastingDataset(
        zarr_path="path/to/your/data.zarr", input_frames=4, target_frames=18
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        input_frames, target_frames = dataset[0]
        print(f"Input shape: {input_frames.shape}")
        print(f"Target shape: {target_frames.shape}")
