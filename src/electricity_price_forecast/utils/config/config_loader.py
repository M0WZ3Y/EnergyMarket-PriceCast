"""
Configuration Loader for Electricity Price Forecasting Project

This module provides utilities for loading and managing configuration files
for the electricity price forecasting project.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader class for managing project configurations.
    
    This class handles loading YAML configuration files and provides
    easy access to configuration parameters throughout the project.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir (str): Path to the configuration directory.
                            Defaults to 'config/'
        """
        if config_dir is None:
            # Get the project root directory (go up from src/electricity_price_forecast/utils/config/)
            self.project_root = Path(__file__).parent.parent.parent.parent.parent
            config_dir = self.project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            config_name (str): Name of the configuration file (without .yaml extension)
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # Cache the configuration
            self.configs[config_name] = config
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get a configuration, loading it if not already cached.
        
        Args:
            config_name (str): Name of the configuration
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if config_name not in self.configs:
            self.load_config(config_name)
        
        return self.configs[config_name]
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        return self.get_config('model_config')
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Dict[str, Any]: Data configuration dictionary
        """
        return self.get_config('data_config')
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        Get experiment configuration.
        
        Returns:
            Dict[str, Any]: Experiment configuration dictionary
        """
        return self.get_config('experiment_config')
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """
        Update a configuration with new values.
        
        Args:
            config_name (str): Name of the configuration to update
            updates (Dict[str, Any]): Dictionary of updates to apply
        """
        if config_name not in self.configs:
            self.load_config(config_name)
        
        # Deep update the configuration
        self._deep_update(self.configs[config_name], updates)
    
    def save_config(self, config_name: str) -> None:
        """
        Save a configuration to file.
        
        Args:
            config_name (str): Name of the configuration to save
        """
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not loaded")
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.configs[config_name], file, default_flow_style=False, indent=2)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Deep update a dictionary with another dictionary.
        
        Args:
            base_dict (Dict[str, Any]): Base dictionary to update
            update_dict (Dict[str, Any]): Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def list_configs(self) -> list:
        """
        List all available configuration files.
        
        Returns:
            list: List of configuration file names (without .yaml extension)
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)
        
        return config_files
    
    def validate_config(self, config_name: str) -> bool:
        """
        Validate a configuration file.
        
        Args:
            config_name (str): Name of the configuration to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            config = self.get_config(config_name)
            
            # Basic validation - check if config is a dictionary
            if not isinstance(config, dict):
                return False
            
            # Add more specific validation logic here based on config type
            if config_name == 'model_config':
                return self._validate_model_config(config)
            elif config_name == 'data_config':
                return self._validate_data_config(config)
            
            return True
            
        except Exception:
            return False
    
    def _validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_sections = ['models', 'hyperparameter_tuning', 'metrics']
        
        for section in required_sections:
            if section not in config:
                return False
        
        return True
    
    def _validate_data_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate data configuration.
        
        Args:
            config (Dict[str, Any]): Data configuration
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_sections = ['data_sources', 'preprocessing', 'feature_engineering']
        
        for section in required_sections:
            if section not in config:
                return False
        
        return True


# Global configuration loader instance
config_loader = ConfigLoader()

# Convenience functions
def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return config_loader.get_model_config()

def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return config_loader.get_data_config()

def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration."""
    return config_loader.get_experiment_config()