#!/usr/bin/env python3
"""
Script to collect Settlements Verified Hourly LMPs data for 2014.

This script downloads the rt_da_monthly_lmps data which contains
verified hourly Real-Time LMPs for aggregate and zonal pnodes used in settlements.
This is the most accurate historical price data available.
"""

import os
import subprocess
import pandas as pd
from datetime import datetime
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('settlement_lmp_2014_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_settlement_lmp_data():
    """
    Collect Settlements Verified Hourly LMPs data.
    
    Returns:
        str: Path to the collected data file
    """
    logger.info("Starting collection of Settlements Verified Hourly LMPs data")
    
    # Create output directory
    output_dir = "02_data/raw/pjm/settlement_verified_hourly_lmps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file
    output_file = os.path.join(output_dir, "settlement_verified_lmps_2014.csv")
    
    try:
        # Build command to collect data
        cmd = (
            f'cd pjm_dataminer-master && set PYTHONIOENCODING=utf-8 && '
            f'venv\\Scripts\\activate && python fetch_pjm.py '
            f'-u rt_da_monthly_lmps -f csv -o ..\\{output_file}'
        )
        
        logger.info("Running command to collect Settlements Verified Hourly LMPs data...")
        logger.info(f"Output file: {output_file}")
        
        # Execute command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            # Check file size
            file_size = os.path.getsize(output_file)
            if file_size > 0:
                logger.info(f"SUCCESS: Data collected successfully!")
                logger.info(f"File: {output_file}")
                logger.info(f"File size: {file_size/1024/1024:.2f} MB")
                return output_file
            else:
                logger.error("FAILED: File is empty")
                return None
        else:
            logger.error("FAILED: Command failed or file not created")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("FAILED: Command timed out")
        return None
    except Exception as e:
        logger.error(f"FAILED: Error occurred: {str(e)}")
        return None

def analyze_and_filter_data(file_path, target_year=2014):
    """
    Analyze the collected data and filter for 2014.
    
    Args:
        file_path (str): Path to the collected data file
        target_year (int): Target year to filter for
    
    Returns:
        str: Path to the filtered file
    """
    logger.info(f"Analyzing and filtering {file_path} for {target_year} data")
    
    try:
        # Read the CSV file
        logger.info("Reading the collected data...")
        df = pd.read_csv(file_path)
        total_rows = len(df)
        logger.info(f"Total rows in dataset: {total_rows}")
        
        if total_rows == 0:
            logger.error("Dataset is empty")
            return None
        
        # Show column information
        logger.info(f"Columns: {list(df.columns)}")
        
        # Try to identify date/time column
        date_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                date_columns.append(col)
        
        if not date_columns:
            logger.error("No date column found in the dataset")
            return None
        
        # Use the first date column found
        date_col = date_columns[0]
        logger.info(f"Using date column: {date_col}")
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Show date range
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        logger.info(f"Date range in dataset: {min_date} to {max_date}")
        
        # Filter for target year
        logger.info(f"Filtering for {target_year} data...")
        filtered_df = df[df[date_col].dt.year == target_year]
        filtered_rows = len(filtered_df)
        logger.info(f"Rows from {target_year}: {filtered_rows}")
        
        if filtered_rows == 0:
            logger.warning(f"No data found for {target_year}")
            return file_path
        
        # Show 2014 date range
        min_2014 = filtered_df[date_col].min()
        max_2014 = filtered_df[date_col].max()
        logger.info(f"{target_year} data range: {min_2014} to {max_2014}")
        
        # Create filtered filename
        base_name = os.path.basename(file_path)
        filtered_filename = f"filtered_{target_year}_{base_name}"
        filtered_path = os.path.join(os.path.dirname(file_path), filtered_filename)
        
        # Save filtered data
        logger.info(f"Saving filtered data to: {filtered_path}")
        filtered_df.to_csv(filtered_path, index=False)
        
        # Get file size
        filtered_size = os.path.getsize(filtered_path) / (1024 * 1024)  # MB
        logger.info(f"Filtered file size: {filtered_size:.2f} MB")
        
        return filtered_path
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return file_path

def main():
    """Main function to orchestrate the data collection process."""
    logger.info("=" * 80)
    logger.info("SETTLEMENTS VERIFIED HOURLY LMPs 2014 DATA COLLECTION")
    logger.info("=" * 80)
    logger.info("This script collects verified hourly Real-Time LMPs used in settlements")
    logger.info("This is the most accurate historical price data available")
    logger.info("=" * 80)
    
    # Step 1: Collect the data
    logger.info("Step 1: Collecting Settlements Verified Hourly LMPs data...")
    data_file = collect_settlement_lmp_data()
    
    if not data_file:
        logger.error("FAILED: Could not collect data")
        return False
    
    # Step 2: Analyze and filter for 2014
    logger.info("Step 2: Analyzing and filtering for 2014 data...")
    filtered_file = analyze_and_filter_data(data_file, target_year=2014)
    
    if filtered_file:
        logger.info("=" * 80)
        logger.info("SUCCESS: Settlements Verified Hourly LMPs 2014 data collection completed!")
        logger.info(f"Final dataset: {filtered_file}")
        logger.info("=" * 80)
        
        # Summary
        try:
            df = pd.read_csv(filtered_file)
            logger.info(f"Final dataset summary:")
            logger.info(f"- Total rows: {len(df)}")
            logger.info(f"- Columns: {list(df.columns)}")
            logger.info(f"- File size: {os.path.getsize(filtered_file)/1024/1024:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not generate summary: {str(e)}")
        
        return True
    else:
        logger.error("FAILED: Could not filter data")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)