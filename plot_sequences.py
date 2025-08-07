#!/usr/bin/env python3
"""
Script to visualize input and target sequences from Zarr dataset.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add the data directory to path to import the dataset
data_path = Path(__file__).parent / "data"
sys.path.insert(0, str(data_path))

try:
    from zarr_dataset import ZarrNowcastingDataset, create_train_val_split
except ImportError:
    print("❌ Could not import zarr_dataset. Make sure data/zarr_dataset.py exists.")
    sys.exit(1)


def plot_sequence(input_frames, target_frames, sample_idx=0, save_path=None):
    """
    Plot input and target sequences side by side.

    Args:
        input_frames: Input tensor of shape (input_frames, channels, height, width)
        target_frames: Target tensor of shape (target_frames, channels, height, width)
        sample_idx: Index of the sample being plotted
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(input_frames, torch.Tensor):
        input_frames = input_frames.numpy()
    if isinstance(target_frames, torch.Tensor):
        target_frames = target_frames.numpy()

    # Remove channel dimension if it's 1
    if input_frames.shape[1] == 1:
        input_frames = input_frames[:, 0, :, :]  # (time, height, width)
    if target_frames.shape[1] == 1:
        target_frames = target_frames[:, 0, :, :]  # (time, height, width)

    num_input = input_frames.shape[0]
    num_target = target_frames.shape[0]
    total_frames = num_input + num_target

    # Create subplot grid
    cols = min(6, total_frames)  # Max 6 columns
    rows = (total_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot input frames
    for i in range(num_input):
        row = i // cols
        col = i % cols

        im = axes[row, col].imshow(input_frames[i], cmap="viridis", origin="lower")
        axes[row, col].set_title(f"Input {i + 1}", fontsize=10)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    # Plot target frames (show only first few to avoid clutter)
    frames_to_show = min(num_target, total_frames - num_input)
    for i in range(frames_to_show):
        frame_idx = num_input + i
        row = frame_idx // cols
        col = frame_idx % cols

        if row < rows and col < cols:
            im = axes[row, col].imshow(target_frames[i], cmap="viridis", origin="lower")
            axes[row, col].set_title(f"Target {i + 1}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(total_frames, rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row, col].set_visible(False)

    plt.suptitle(
        f"Sample {sample_idx}: Input and Target Sequences\n"
        f"Input: {num_input} frames, Target: {num_target} frames",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig


def plot_sequence_timeline(input_frames, target_frames, sample_idx=0, save_path=None):
    """
    Plot input and target sequences in a timeline format.

    Args:
        input_frames: Input tensor of shape (input_frames, channels, height, width)
        target_frames: Target tensor of shape (target_frames, channels, height, width)
        sample_idx: Index of the sample being plotted
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(input_frames, torch.Tensor):
        input_frames = input_frames.numpy()
    if isinstance(target_frames, torch.Tensor):
        target_frames = target_frames.numpy()

    # Remove channel dimension if it's 1
    if input_frames.shape[1] == 1:
        input_frames = input_frames[:, 0, :, :]  # (time, height, width)
    if target_frames.shape[1] == 1:
        target_frames = target_frames[:, 0, :, :]  # (time, height, width)

    num_input = input_frames.shape[0]
    num_target = target_frames.shape[0]

    # Create timeline plot
    fig, axes = plt.subplots(
        2, max(num_input, num_target), figsize=(2 * max(num_input, num_target), 6)
    )

    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # Plot input frames
    for i in range(num_input):
        im = axes[0, i].imshow(input_frames[i], cmap="viridis", origin="lower")
        axes[0, i].set_title(f"Input t-{num_input - i}", fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    # Hide unused input subplots
    for i in range(num_input, axes.shape[1]):
        axes[0, i].set_visible(False)

    # Plot target frames
    for i in range(min(num_target, axes.shape[1])):
        im = axes[1, i].imshow(target_frames[i], cmap="viridis", origin="lower")
        axes[1, i].set_title(f"Target t+{i + 1}", fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    # Hide unused target subplots
    for i in range(min(num_target, axes.shape[1]), axes.shape[1]):
        axes[1, i].set_visible(False)

    # Add row labels
    axes[0, 0].set_ylabel("Input\nSequence", fontsize=12, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("Target\nSequence", fontsize=12, rotation=0, labelpad=50)

    plt.suptitle(
        f"Sample {sample_idx}: Temporal Sequence\n"
        f"Input: {num_input} frames → Target: {num_target} frames",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Timeline plot saved to: {save_path}")

    return fig


def plot_statistics(input_frames, target_frames, sample_idx=0, save_path=None):
    """
    Plot statistics of the input and target sequences.

    Args:
        input_frames: Input tensor of shape (input_frames, channels, height, width)
        target_frames: Target tensor of shape (target_frames, channels, height, width)
        sample_idx: Index of the sample being plotted
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(input_frames, torch.Tensor):
        input_frames = input_frames.numpy()
    if isinstance(target_frames, torch.Tensor):
        target_frames = target_frames.numpy()

    # Remove channel dimension if it's 1
    if input_frames.shape[1] == 1:
        input_frames = input_frames[:, 0, :, :]
    if target_frames.shape[1] == 1:
        target_frames = target_frames[:, 0, :, :]

    # Calculate statistics
    input_means = [frame.mean() for frame in input_frames]
    input_stds = [frame.std() for frame in input_frames]
    input_maxs = [frame.max() for frame in input_frames]
    input_mins = [frame.min() for frame in input_frames]

    target_means = [frame.mean() for frame in target_frames]
    target_stds = [frame.std() for frame in target_frames]
    target_maxs = [frame.max() for frame in target_frames]
    target_mins = [frame.min() for frame in target_frames]

    # Create time axis
    input_times = list(range(-len(input_frames), 0))
    target_times = list(range(1, len(target_frames) + 1))
    all_times = input_times + target_times
    all_means = input_means + target_means
    all_stds = input_stds + target_stds
    all_maxs = input_maxs + target_maxs
    all_mins = input_mins + target_mins

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mean values
    axes[0, 0].plot(input_times, input_means, "bo-", label="Input", linewidth=2)
    axes[0, 0].plot(target_times, target_means, "ro-", label="Target", linewidth=2)
    axes[0, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Mean Values Over Time")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Mean Value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Standard deviation
    axes[0, 1].plot(input_times, input_stds, "bo-", label="Input", linewidth=2)
    axes[0, 1].plot(target_times, target_stds, "ro-", label="Target", linewidth=2)
    axes[0, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Standard Deviation Over Time")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Standard Deviation")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Min/Max values
    axes[1, 0].plot(all_times, all_maxs, "g^-", label="Max", linewidth=2)
    axes[1, 0].plot(all_times, all_mins, "rv-", label="Min", linewidth=2)
    axes[1, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("Min/Max Values Over Time")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram of all values
    all_input_values = input_frames.flatten()
    all_target_values = target_frames.flatten()

    axes[1, 1].hist(all_input_values, bins=50, alpha=0.7, label="Input", density=True)
    axes[1, 1].hist(all_target_values, bins=50, alpha=0.7, label="Target", density=True)
    axes[1, 1].set_title("Value Distribution")
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Sample {sample_idx}: Statistical Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Statistics plot saved to: {save_path}")

    return fig


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize Zarr dataset sequences")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="data/test_dataset.zarr",
        help="Path to the Zarr dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["grid", "timeline", "stats", "all"],
        default="all",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )

    args = parser.parse_args()

    # Check if dataset exists
    if not Path(args.zarr_path).exists():
        print(f"❌ Dataset not found: {args.zarr_path}")
        print("Create a dataset first using create_dummy_zarr.py")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading dataset from: {args.zarr_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    try:
        # Create train/val split
        train_indices, val_indices = create_train_val_split(
            args.zarr_path, train_ratio=0.8, seed=42
        )

        # Create dataset
        dataset = ZarrNowcastingDataset(
            zarr_path=args.zarr_path,
            input_frames=4,
            target_frames=18,
            indices=train_indices[: args.num_samples],
        )

        print(f"Dataset loaded with {len(dataset)} samples")

        # Generate plots for each sample
        for i in range(min(args.num_samples, len(dataset))):
            print(
                f"\nProcessing sample {i + 1}/{min(args.num_samples, len(dataset))}..."
            )

            input_frames, target_frames = dataset[i]

            print(f"  Input shape: {input_frames.shape}")
            print(f"  Target shape: {target_frames.shape}")
            print(
                f"  Value range: [{input_frames.min():.3f}, {input_frames.max():.3f}] → "
                f"[{target_frames.min():.3f}, {target_frames.max():.3f}]"
            )

            # Generate different types of plots
            if args.plot_type in ["grid", "all"]:
                save_path = output_dir / f"sample_{i:03d}_grid.png"
                fig = plot_sequence(input_frames, target_frames, i, save_path)
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)

            if args.plot_type in ["timeline", "all"]:
                save_path = output_dir / f"sample_{i:03d}_timeline.png"
                fig = plot_sequence_timeline(input_frames, target_frames, i, save_path)
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)

            if args.plot_type in ["stats", "all"]:
                save_path = output_dir / f"sample_{i:03d}_stats.png"
                fig = plot_statistics(input_frames, target_frames, i, save_path)
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)

        print(
            f"\n✅ Successfully generated plots for {min(args.num_samples, len(dataset))} samples"
        )
        print(f"📁 Plots saved in: {output_dir}")

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
