"""Configuration utilities for DGMR training."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration class for DGMR training."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict

        # Create nested attribute access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        return self._config

    def update(self, other_config: Dict[str, Any]) -> None:
        """Update config with another dictionary."""

        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self._config, other_config)

        # Recreate attributes
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with nested attribute access
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for config-based training."""
    parser = argparse.ArgumentParser(description="Train DGMR on Zarr dataset")

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--zarr_path", type=str, help="Override zarr_path from config")
    parser.add_argument(
        "--batch_size", type=int, help="Override batch_size from config"
    )
    parser.add_argument(
        "--max_epochs", type=int, help="Override max_epochs from config"
    )
    parser.add_argument(
        "--gen_lr", type=float, help="Override generator learning rate from config"
    )
    parser.add_argument(
        "--disc_lr", type=float, help="Override discriminator learning rate from config"
    )
    parser.add_argument(
        "--devices", type=int, help="Override number of devices from config"
    )
    parser.add_argument(
        "--accelerator", type=str, help="Override accelerator type from config"
    )

    return parser.parse_args()


def create_config_from_args() -> Config:
    """
    Create configuration by loading YAML and applying command line overrides.

    Returns:
        Config object with all settings
    """
    args = parse_args()

    # Load base config
    config = load_config(args.config)

    # Apply command line overrides
    overrides = {}

    if args.zarr_path is not None:
        overrides.setdefault("data", {})["zarr_path"] = args.zarr_path

    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size

    if args.max_epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.max_epochs

    if args.gen_lr is not None:
        overrides.setdefault("model", {})["gen_lr"] = args.gen_lr

    if args.disc_lr is not None:
        overrides.setdefault("model", {})["disc_lr"] = args.disc_lr

    if args.devices is not None:
        overrides.setdefault("hardware", {})["devices"] = args.devices

    if args.accelerator is not None:
        overrides.setdefault("hardware", {})["accelerator"] = args.accelerator

    # Update config with overrides
    if overrides:
        config.update(overrides)
        print("Applied command line overrides:")
        for section, values in overrides.items():
            for key, value in values.items():
                print(f"  {section}.{key} = {value}")

    return config


def print_config(config: Config) -> None:
    """Print configuration in a readable format."""

    def print_section(
        section_name: str, section_config: Config, indent: int = 0
    ) -> None:
        prefix = "  " * indent
        print(f"{prefix}{section_name}:")

        for key, value in section_config.to_dict().items():
            if isinstance(value, dict):
                print_section(key, Config(value), indent + 1)
            else:
                print(f"{prefix}  {key}: {value}")

    print("Configuration:")
    print("=" * 50)

    for section_name, section_value in config.to_dict().items():
        if isinstance(section_value, dict):
            print_section(section_name, Config(section_value))
        else:
            print(f"{section_name}: {section_value}")
        print()


if __name__ == "__main__":
    # Example usage
    config = create_config_from_args()
    print_config(config)
