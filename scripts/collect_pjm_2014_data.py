#!/usr/bin/env python3
"""
PJM Data Collection Script for 2014 Historical Data

This script collects the key PJM market data needed for electricity price forecasting
research, focusing on data from 2014 as specified in the thesis requirements.

Key Data Types to Collect:
1. Real-Time Hourly LMPs - Primary target variable for hourly forecasting
2. Day-Ahead Hourly LMPs - Key feature for price forecasting models  
3. Settlements Verified Hourly LMPs - Most accurate historical price data
4. Hourly Load Data - Essential feature for price forecasting
5. Generation by Fuel Type - Important for understanding supply dynamics
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
code_path = project_root / '03_code'
sys.path.insert(0, str(code_path))

from data_pipeline.data_collection.pjm_data_collector import PJMDataCollector
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_2014_pjm_data():
    """
    Collect PJM data for 2014 research period.
    
    Note: The PJM API may not support direct year filtering, so we collect all available
    historical data and document the intended research period.
    """
    
    # Initialize the collector
    collector = PJMDataCollector()
    
    # Define the key data types for electricity price forecasting
    key_data_types = [
        {
            'type': 'rt_da_monthly_lmps',
            'description': 'Settlements Verified Hourly LMPs - Most accurate historical price data',
            'priority': 'HIGH'
        },
        {
            'type': 'da_hrl_lmps', 
            'description': 'Day-Ahead Hourly LMPs - Key feature for price forecasting models',
            'priority': 'HIGH'
        },
        {
            'type': 'hrl_load_metered',
            'description': 'Hourly Load: Metered - Actual load consumption data',
            'priority': 'HIGH'
        },
        {
            'type': 'gen_by_fuel',
            'description': 'Generation by Fuel Type - Fuel mix of generation resources',
            'priority': 'MEDIUM'
        },
        {
            'type': 'solar_gen',
            'description': 'Solar Generation - Hourly solar generation amounts',
            'priority': 'MEDIUM'
        },
        {
            'type': 'wind_gen',
            'description': 'Wind Generation - Hourly wind generation amounts', 
            'priority': 'MEDIUM'
        }
    ]
    
    # Output directory for 2014 data
    output_dir = project_root / '02_data' / 'raw' / 'pjm' / '2014_historical'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PJM data collection for 2014 research period")
    logger.info(f"Output directory: {output_dir}")
    
    collected_files = []
    
    # Collect each data type
    for data_info in key_data_types:
        data_type = data_info['type']
        description = data_info['description']
        priority = data_info['priority']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting: {data_type}")
        logger.info(f"Description: {description}")
        logger.info(f"Priority: {priority}")
        logger.info(f"{'='*60}")
        
        try:
            # Collect historical data (2014-2024 range for documentation)
            file_paths = collector.collect_historical_data(
                data_type=data_type,
                start_year=2014,
                end_year=2024,
                output_dir=str(output_dir)
            )
            
            collected_files.extend(file_paths)
            logger.info(f"✅ Successfully collected {data_type}")
            
            # Preview the data structure
            if file_paths:
                try:
                    preview = collector.preview_data(file_paths[0], num_rows=3)
                    logger.info(f"Data preview for {data_type}:")
                    logger.info(f"Columns: {list(preview.columns)}")
                    logger.info(f"Sample data shape: {preview.shape}")
                except Exception as e:
                    logger.warning(f"Could not preview data for {data_type}: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect {data_type}: {str(e)}")
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PJM DATA COLLECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files collected: {len(collected_files)}")
    
    for file_path in collected_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"  - {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    logger.info(f"\nAll data saved to: {output_dir}")
    logger.info("Note: The collected data contains all available historical data.")
    logger.info("You will need to filter for 2014 data during preprocessing.")
    
    return collected_files

def main():
    """Main execution function."""
    try:
        collected_files = collect_2014_pjm_data()
        
        if collected_files:
            print(f"\n✅ Successfully collected {len(collected_files)} data files!")
            print("Next steps:")
            print("1. Check the collected data in the output directory")
            print("2. Filter for 2014 data during preprocessing")
            print("3. Use the data for your electricity price forecasting models")
        else:
            print("\n❌ No data files were collected. Please check the logs for errors.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Data collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Data collection failed: {str(e)}")
        logger.exception("Data collection failed")

if __name__ == "__main__":
    main()