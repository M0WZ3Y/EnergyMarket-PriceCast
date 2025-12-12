"""
PJM Data Collector Module

This module provides an interface to collect data from PJM Data Miner API
for electricity market forecasting research.

Key Data Types Available:
- Real-Time Hourly LMPs (rt_hrl_lmps)
- Day-Ahead Hourly LMPs (da_hrl_lmps) 
- Settlements Verified Hourly LMPs (rt_da_monthly_lmps)
- Hourly Load: Metered (hrl_load_metered)
- Solar Generation (solar_gen)
- Wind Generation (wind_gen)
- Generation by Fuel Type (gen_by_fuel)
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# Add the PJM dataminer path to sys.path
PJM_DATAMINER_PATH = os.path.join(os.path.dirname(__file__), '../../../../pjm_dataminer-master')
if PJM_DATAMINER_PATH not in sys.path:
    sys.path.append(PJM_DATAMINER_PATH)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PJMDataCollector:
    """
    Interface for collecting PJM market data using the PJM dataminer tool.
    """
    
    def __init__(self, pjm_dataminer_path: str = None):
        """
        Initialize the PJM Data Collector.
        
        Args:
            pjm_dataminer_path: Path to the PJM dataminer directory
        """
        self.pjm_dataminer_path = pjm_dataminer_path or PJM_DATAMINER_PATH
        self.venv_activate = os.path.join(self.pjm_dataminer_path, 'venv', 'Scripts', 'activate')
        self.fetch_script = os.path.join(self.pjm_dataminer_path, 'fetch_pjm.py')
        
        # Available data types with descriptions
        self.available_data_types = {
            'rt_hrl_lmps': 'Real-Time Hourly LMPs - Primary target variable for hourly forecasting',
            'da_hrl_lmps': 'Day-Ahead Hourly LMPs - Key feature for price forecasting models', 
            'rt_da_monthly_lmps': 'Settlements Verified Hourly LMPs - Most accurate historical price data',
            'hrl_load_metered': 'Hourly Load: Metered - Actual load consumption data',
            'solar_gen': 'Solar Generation - Hourly solar generation amounts',
            'wind_gen': 'Wind Generation - Hourly wind generation amounts',
            'gen_by_fuel': 'Generation by Fuel Type - Fuel mix of generation resources',
            'load_frcstd_7_day': 'Seven-Day Load Forecast - PJM hourly load forecast',
            'rt_hrl_lmps': 'Real-Time Hourly LMPs - Real-time locational marginal pricing',
            'da_hrl_lmps': 'Day-Ahead Hourly LMPs - Day-ahead locational marginal pricing'
        }
    
    def collect_data(self, 
                    data_type: str, 
                    output_format: str = 'csv',
                    output_file: str = None,
                    custom_output_dir: str = None) -> str:
        """
        Collect data from PJM API.
        
        Args:
            data_type: Type of data to collect (e.g., 'rt_hrl_lmps', 'da_hrl_lmps')
            output_format: Output format ('csv', 'json', 'xls')
            output_file: Custom output filename
            custom_output_dir: Custom output directory
            
        Returns:
            Path to the downloaded file
            
        Raises:
            ValueError: If data_type is not available
            RuntimeError: If data collection fails
        """
        if data_type not in self.available_data_types:
            raise ValueError(f"Data type '{data_type}' not available. Available types: {list(self.available_data_types.keys())}")
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{data_type}_{timestamp}.{output_format}"
        
        # Set output directory
        if custom_output_dir:
            os.makedirs(custom_output_dir, exist_ok=True)
            output_path = os.path.join(custom_output_dir, output_file)
        else:
            # Default to project data directory
            output_dir = os.path.join(os.path.dirname(__file__), '../../../02_data/raw/pjm')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_file)
        
        # Build command
        cmd = [
            'cd', self.pjm_dataminer_path, '&&',
            'set', 'PYTHONIOENCODING=utf-8', '&&',
            self.venv_activate, '&&',
            'python', self.fetch_script,
            '-u', data_type,
            '-f', output_format,
            '-o', output_path
        ]
        
        logger.info(f"Collecting {data_type} data...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute command
        try:
            result = subprocess.run(
                ' '.join(cmd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError(f"Failed to collect {data_type} data: {result.stderr}")
            
            logger.info(f"Successfully collected {data_type} data to {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Data collection for {data_type} timed out after 1 hour")
        except Exception as e:
            raise RuntimeError(f"Failed to collect {data_type} data: {str(e)}")
    
    def collect_historical_data(self,
                               data_type: str,
                               start_year: int = 2014,
                               end_year: int = 2024,
                               output_dir: str = None) -> List[str]:
        """
        Collect historical data for multiple years.
        Note: PJM API may not support direct year filtering, so this will collect
        all available data and the user may need to filter post-download.
        
        Args:
            data_type: Type of data to collect
            start_year: Starting year (for documentation purposes)
            end_year: Ending year (for documentation purposes)
            output_dir: Output directory for collected data
            
        Returns:
            List of paths to downloaded files
        """
        logger.info(f"Collecting {data_type} data for period {start_year}-{end_year}")
        
        # For now, PJM API doesn't seem to support year-based filtering
        # We'll collect all available data and document the intended period
        output_file = f"{data_type}_historical_{start_year}_{end_year}.csv"
        
        try:
            file_path = self.collect_data(
                data_type=data_type,
                output_file=output_file,
                custom_output_dir=output_dir
            )
            return [file_path]
        except Exception as e:
            logger.error(f"Failed to collect historical data for {data_type}: {str(e)}")
            raise
    
    def get_available_data_types(self) -> Dict[str, str]:
        """Get list of available data types with descriptions."""
        return self.available_data_types.copy()
    
    def preview_data(self, file_path: str, num_rows: int = 5) -> pd.DataFrame:
        """
        Preview downloaded data.
        
        Args:
            file_path: Path to the data file
            num_rows: Number of rows to preview
            
        Returns:
            DataFrame with preview of the data
        """
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, nrows=num_rows)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path, nrows=num_rows)
        else:
            raise ValueError(f"Unsupported file format for preview: {file_path}")

def main():
    """
    Example usage of PJM Data Collector
    """
    collector = PJMDataCollector()
    
    # Print available data types
    print("Available PJM Data Types:")
    for data_type, description in collector.get_available_data_types().items():
        print(f"  {data_type}: {description}")
    
    # Example: Collect Day-Ahead Hourly LMPs data
    try:
        print("\nCollecting Day-Ahead Hourly LMPs data...")
        output_path = collector.collect_data('da_hrl_lmps')
        print(f"Data saved to: {output_path}")
        
        # Preview the data
        print("\nData preview:")
        preview = collector.preview_data(output_path)
        print(preview.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()