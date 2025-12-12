#!/usr/bin/env python3
"""
Multi-Year PJM LMP Data Collection Script
Collects Day-Ahead Hourly LMPs and Settlements Verified Hourly LMPs for years 2014-2024
Each year is stored in separate files to keep datasets intact
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
import logging
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_year_pjm_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiYearPJMCollector:
    """Multi-year PJM data collector for LMP data"""
    
    def __init__(self):
        self.pjm_dir = os.path.join(project_root, 'pjm_dataminer-master')
        self.data_dir = os.path.join(project_root, '02_data', 'raw', 'pjm')
        self.years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        
        # Dataset configurations
        self.datasets = {
            'da_hrl_lmps': {
                'name': 'Day-Ahead Hourly LMPs',
                'description': 'Key feature for price forecasting models',
                'endpoint': 'da_hrl_lmps',
                'subdir': 'day_ahead_hourly_lmps'
            },
            'rt_da_monthly_lmps': {
                'name': 'Settlements Verified Hourly LMPs',
                'description': 'Most accurate historical price data',
                'endpoint': 'rt_da_monthly_lmps',
                'subdir': 'settlement_verified_hourly_lmps'
            }
        }
        
        logger.info("Multi-Year PJM Data Collector initialized")
        logger.info(f"Project root: {project_root}")
        logger.info(f"PJM directory: {self.pjm_dir}")
        logger.info(f"Data output directory: {self.data_dir}")
        logger.info(f"Years to collect: {self.years}")
        logger.info(f"Datasets: {list(self.datasets.keys())}")
    
    def create_directories(self):
        """Create necessary directories for all years and datasets"""
        for dataset_key, dataset_info in self.datasets.items():
            for year in self.years:
                year_dir = os.path.join(self.data_dir, dataset_info['subdir'], str(year))
                os.makedirs(year_dir, exist_ok=True)
        logger.info("Created all necessary directories")
    
    def run_pjm_command(self, endpoint, output_file):
        """Run PJM dataminer command"""
        # Properly quote the output file path to handle spaces
        quoted_output_file = f'"{output_file}"'
        
        command_str = f'cd "{self.pjm_dir}" && set PYTHONIOENCODING=utf-8 && venv\\Scripts\\activate && python fetch_pjm.py -u {endpoint} -f csv -o {quoted_output_file}'
        logger.info(f"Executing: {command_str}")
        
        try:
            result = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.pjm_dir
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded data from {endpoint}")
                return True
            else:
                logger.error(f"Error downloading data from {endpoint}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception running PJM command for {endpoint}: {str(e)}")
            return False
    
    def process_and_filter_year_data(self, input_file, output_file, year, dataset_info):
        """Process downloaded data and filter for specific year"""
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        logger.info(f"Processing {dataset_info['name']} data for year {year}")
        
        try:
            # Get file size
            file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
            # Read in chunks to handle large file
            chunk_size = 50000
            chunks = []
            total_rows = 0
            
            logger.info("Reading data in chunks...")
            for chunk in pd.read_csv(input_file, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows % 10000 == 0:
                    logger.info(f"Read {total_rows:,} rows...")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total records in file: {len(df):,}")
            
            # Check date column and filter for specific year
            if 'datetime_beginning_ept' in df.columns:
                df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
                df_year = df[df['datetime_beginning_ept'].dt.year == year]
                
                logger.info(f"Records from {year}: {len(df_year):,}")
                
                if len(df_year) == 0:
                    logger.warning(f"No records found for year {year}")
                    return False
                
                # Save year-specific data
                df_year.to_csv(output_file, index=False)
                logger.info(f"{year} {dataset_info['name']} data saved to: {output_file}")
                
                # Save summary
                summary_file = os.path.join(os.path.dirname(output_file), f'{dataset_info["endpoint"]}_{year}_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write(f"{dataset_info['name']} Data Summary - {year}\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Total records downloaded: {len(df):,}\n")
                    f.write(f"Records from {year}: {len(df_year):,}\n")
                    f.write(f"Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}\n")
                    f.write(f"{year} date range: {df_year['datetime_beginning_ept'].min()} to {df_year['datetime_beginning_ept'].max()}\n")
                    f.write(f"\nColumns: {list(df.columns)}\n")
                    
                    # Analyze market run types if available
                    if 'market_run_type' in df.columns:
                        market_types = df['market_run_type'].value_counts()
                        f.write(f"\nMarket Run Types (Complete Dataset):\n")
                        for market_type, count in market_types.items():
                            f.write(f"  {market_type}: {count:,} records\n")
                        
                        market_types_year = df_year['market_run_type'].value_counts()
                        f.write(f"\nMarket Run Types ({year} Only):\n")
                        for market_type, count in market_types_year.items():
                            f.write(f"  {market_type}: {count:,} records\n")
                
                logger.info(f"Summary saved to: {summary_file}")
                return True
                
            else:
                logger.warning("Date column not found in data")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {dataset_info['name']} data for {year}: {str(e)}")
            return False
    
    def collect_dataset_for_year(self, dataset_key, year):
        """Collect a specific dataset for a specific year"""
        dataset_info = self.datasets[dataset_key]
        
        logger.info("=" * 80)
        logger.info(f"COLLECTING {dataset_info['name'].upper()} FOR YEAR {year}")
        logger.info("=" * 80)
        
        # Create temporary file for download
        temp_file = os.path.join(self.pjm_dir, f'temp_{dataset_key}_{year}.csv')
        
        # Download data
        logger.info(f"Downloading {dataset_info['name']} data...")
        success = self.run_pjm_command(dataset_info['endpoint'], temp_file)
        
        if not success:
            logger.error(f"Failed to download {dataset_info['name']} for year {year}")
            return False
        
        # Process and filter data
        year_dir = os.path.join(self.data_dir, dataset_info['subdir'], str(year))
        output_file = os.path.join(year_dir, f'{dataset_key}_{year}.csv')
        
        success = self.process_and_filter_year_data(temp_file, output_file, year, dataset_info)
        
        # Clean up temporary file
        try:
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")
        except:
            pass
        
        return success
    
    def collect_all_years_for_dataset(self, dataset_key):
        """Collect data for all years for a specific dataset"""
        dataset_info = self.datasets[dataset_key]
        logger.info(f"Starting collection of {dataset_info['name']} for all years...")
        
        success_count = 0
        for year in self.years:
            logger.info(f"Processing year {year}...")
            if self.collect_dataset_for_year(dataset_key, year):
                success_count += 1
                logger.info(f"Successfully collected {dataset_info['name']} for {year}")
            else:
                logger.error(f"Failed to collect {dataset_info['name']} for {year}")
            
            # Add a small delay between years to avoid overwhelming the API
            time.sleep(2)
        
        logger.info(f"Completed {dataset_info['name']} collection: {success_count}/{len(self.years)} years successful")
        return success_count
    
    def create_master_summary(self):
        """Create master summary of all collected data"""
        logger.info("=" * 80)
        logger.info("CREATING MASTER SUMMARY")
        logger.info("=" * 80)
        
        summary_file = os.path.join(self.data_dir, 'MULTI_YEAR_LMP_DATA_COLLECTION_SUMMARY.md')
        
        with open(summary_file, 'w') as f:
            f.write(f"# Multi-Year PJM LMP Data Collection Summary\n\n")
            f.write(f"**Collection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Years Collected: {', '.join(map(str, self.years))}\n\n")
            f.write(f"## Datasets Collected\n\n")
            
            for dataset_key, dataset_info in self.datasets.items():
                f.write(f"### {dataset_info['name']}\n")
                f.write(f"- **Description:** {dataset_info['description']}\n")
                f.write(f"- **Endpoint:** {dataset_info['endpoint']}\n")
                f.write(f"- **Directory:** `{dataset_info['subdir']}`\n\n")
                
                # Check which years were successfully collected
                f.write(f"**Year-by-Year Status:**\n\n")
                for year in self.years:
                    year_file = os.path.join(self.data_dir, dataset_info['subdir'], str(year), f'{dataset_key}_{year}.csv')
                    if os.path.exists(year_file):
                        file_size = os.path.getsize(year_file) / (1024 * 1024)  # MB
                        f.write(f"- [SUCCESS] {year}: `{dataset_key}_{year}.csv` ({file_size:.2f} MB)\n")
                    else:
                        f.write(f"- [FAILED] {year}: Not collected\n")
                f.write(f"\n")
            
            f.write(f"## Usage\n\n")
            f.write(f"Each year's data is stored in separate files within year-specific directories.\n")
            f.write(f"This structure allows for easy access to specific time periods and\n")
            f.write(f"maintains data integrity across different years.\n\n")
            
            f.write(f"## Directory Structure\n\n")
            f.write(f"```\n")
            f.write(f"02_data/raw/pjm/\n")
            for dataset_key, dataset_info in self.datasets.items():
                f.write(f"├── {dataset_info['subdir']}/\n")
                for year in self.years:
                    f.write(f"│   ├── {year}/\n")
                    f.write(f"│   │   └── {dataset_key}_{year}.csv\n")
            f.write(f"```\n\n")
            
            f.write(f"## Next Steps\n\n")
            f.write(f"1. Data quality validation for each year\n")
            f.write(f"2. Cross-year consistency checks\n")
            f.write(f"3. Feature engineering with temporal considerations\n")
            f.write(f"4. Model development with train/validation/test splits\n")
            f.write(f"5. Performance evaluation across different time periods\n\n")
        
        logger.info(f"Master summary saved to: {summary_file}")
    
    def run_complete_collection(self):
        """Run complete multi-year data collection"""
        logger.info("Starting comprehensive multi-year PJM LMP data collection...")
        
        # Create directories
        self.create_directories()
        
        total_success = 0
        total_datasets = len(self.datasets)
        
        # Collect each dataset for all years
        for dataset_key in self.datasets.keys():
            logger.info(f"Starting collection for dataset: {dataset_key}")
            success_count = self.collect_all_years_for_dataset(dataset_key)
            if success_count > 0:
                total_success += 1
        
        # Create master summary
        self.create_master_summary()
        
        logger.info("=" * 80)
        logger.info("MULTI-YEAR COLLECTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Successfully collected {total_success}/{total_datasets} datasets")
        
        if total_success == total_datasets:
            logger.info("Multi-year PJM LMP data collection completed successfully!")
        else:
            logger.warning("Some datasets may not have been collected completely")
        
        return total_success == total_datasets

def main():
    """Main function"""
    collector = MultiYearPJMCollector()
    success = collector.run_complete_collection()
    
    if success:
        logger.info("Complete multi-year data collection finished successfully!")
        return 0
    else:
        logger.error("Multi-year data collection encountered issues")
        return 1

if __name__ == "__main__":
    exit(main())