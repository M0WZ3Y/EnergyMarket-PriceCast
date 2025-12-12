#!/usr/bin/env python3
"""
Script to collect Real-Time Hourly LMPs data for 2014 in monthly chunks
and combine them into a complete dataset.

This script downloads data for each month separately to avoid timeouts
and then combines all monthly files into a single 2014 dataset.
"""

import os
import sys
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
        logging.FileHandler('rt_lmp_2014_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_monthly_rt_lmp_data(year=2014):
    """
    Collect Real-Time Hourly LMPs data for each month of the specified year.
    
    Args:
        year (int): Year to collect data for (default: 2014)
    
    Returns:
        list: List of monthly file paths
    """
    monthly_files = []
    
    # Define months
    months = [
        ('01', 'January'), ('02', 'February'), ('03', 'March'),
        ('04', 'April'), ('05', 'May'), ('06', 'June'),
        ('07', 'July'), ('08', 'August'), ('09', 'September'),
        ('10', 'October'), ('11', 'November'), ('12', 'December')
    ]
    
    logger.info(f"Starting collection of Real-Time Hourly LMPs data for {year}")
    
    for month_num, month_name in months:
        logger.info(f"Collecting data for {month_name} {year}...")
        
        # Generate output filename for this month
        output_filename = f"rt_hrl_lmps_{year}_{month_num}.csv"
        output_path = f"02_data/raw/pjm/real_time_hourly_lmps/{output_filename}"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Build command to collect data
            cmd = [
                'cd', 'pjm_dataminer-master', '&&',
                'set', 'PYTHONIOENCODING=utf-8', '&&',
                'venv\\Scripts\\activate', '&&',
                'python', 'fetch_pjm.py',
                '-u', 'rt_hrl_lmps',
                '-f', 'csv',
                '-o', f'..\\{output_path}'
            ]
            
            # Execute command
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                'cd pjm_dataminer-master && set PYTHONIOENCODING=utf-8 && venv\\Scripts\\activate && python fetch_pjm.py -u rt_hrl_lmps -f csv -o ..\\' + output_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per month
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                monthly_files.append(output_path)
                logger.info(f"Successfully collected {month_name} {year} data: {output_path}")
                
                # Get file size for logging
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                logger.info(f"File size: {file_size:.2f} MB")
                
            else:
                logger.error(f"Failed to collect {month_name} {year} data")
                logger.error(f"Return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout collecting {month_name} {year} data")
        except Exception as e:
            logger.error(f"Error collecting {month_name} {year} data: {str(e)}")
            continue
        
        # Add delay between requests to avoid overwhelming the API
        time.sleep(5)
    
    logger.info(f"Completed collection. Monthly files: {len(monthly_files)}")
    return monthly_files

def filter_monthly_data_for_year(file_path, target_year=2014):
    """
    Filter monthly data to include only the target year.
    
    Args:
        file_path (str): Path to the monthly CSV file
        target_year (int): Target year to filter for
    
    Returns:
        str: Path to the filtered file
    """
    logger.info(f"Filtering {file_path} for year {target_year}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Original file has {len(df)} rows")
        
        # Try to identify date/time column
        date_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                date_columns.append(col)
        
        if not date_columns:
            logger.warning(f"No date column found in {file_path}")
            return file_path
        
        # Use the first date column found
        date_col = date_columns[0]
        logger.info(f"Using date column: {date_col}")
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filter for target year
        filtered_df = df[df[date_col].dt.year == target_year]
        logger.info(f"Filtered file has {len(filtered_df)} rows for {target_year}")
        
        if len(filtered_df) == 0:
            logger.warning(f"No data found for {target_year} in {file_path}")
            return file_path
        
        # Create filtered filename
        base_name = os.path.basename(file_path)
        name_parts = base_name.replace('.csv', '').split('_')
        filtered_filename = f"{'_'.join(name_parts[:-1])}_filtered_{target_year}.csv"
        filtered_path = os.path.join(os.path.dirname(file_path), filtered_filename)
        
        # Save filtered data
        filtered_df.to_csv(filtered_path, index=False)
        logger.info(f"Saved filtered data to: {filtered_path}")
        
        return filtered_path
        
    except Exception as e:
        logger.error(f"Error filtering {file_path}: {str(e)}")
        return file_path

def combine_monthly_files(monthly_files, output_file):
    """
    Combine multiple monthly CSV files into a single dataset.
    
    Args:
        monthly_files (list): List of monthly file paths
        output_file (str): Path for the combined output file
    
    Returns:
        str: Path to the combined file
    """
    logger.info(f"Combining {len(monthly_files)} monthly files...")
    
    if not monthly_files:
        logger.error("No monthly files to combine")
        return None
    
    try:
        # Read and combine all files
        all_data = []
        
        for i, file_path in enumerate(monthly_files):
            logger.info(f"Processing file {i+1}/{len(monthly_files)}: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                logger.info(f"Added {len(df)} rows from {file_path}")
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No data could be read from monthly files")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} total rows")
        
        # Sort by date if possible
        date_columns = []
        for col in combined_df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                date_columns.append(col)
        
        if date_columns:
            date_col = date_columns[0]
            try:
                combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce')
                combined_df = combined_df.sort_values(date_col)
                logger.info(f"Sorted data by {date_col}")
            except Exception as e:
                logger.warning(f"Could not sort by date: {str(e)}")
        
        # Save combined data
        combined_df.to_csv(output_file, index=False)
        
        # Get file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        logger.info(f"Combined dataset saved to: {output_file}")
        logger.info(f"Combined file size: {file_size:.2f} MB")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error combining monthly files: {str(e)}")
        return None

def main():
    """Main function to orchestrate the data collection process."""
    logger.info("=" * 60)
    logger.info("Starting Real-Time Hourly LMPs 2014 Data Collection")
    logger.info("=" * 60)
    
    # Step 1: Collect monthly data
    logger.info("Step 1: Collecting monthly Real-Time Hourly LMPs data...")
    monthly_files = collect_monthly_rt_lmp_data(year=2014)
    
    if not monthly_files:
        logger.error("No monthly data collected. Exiting.")
        return False
    
    # Step 2: Filter monthly data for 2014
    logger.info("Step 2: Filtering monthly data for 2014...")
    filtered_files = []
    
    for file_path in monthly_files:
        filtered_file = filter_monthly_data_for_year(file_path, target_year=2014)
        filtered_files.append(filtered_file)
    
    # Step 3: Combine filtered files
    logger.info("Step 3: Combining filtered monthly files...")
    output_dir = "02_data/raw/pjm/real_time_hourly_lmps"
    os.makedirs(output_dir, exist_ok=True)
    
    combined_file = os.path.join(output_dir, "rt_hrl_lmps_2014_combined.csv")
    final_file = combine_monthly_files(filtered_files, combined_file)
    
    if final_file:
        logger.info("=" * 60)
        logger.info("SUCCESS: Real-Time Hourly LMPs 2014 data collection completed!")
        logger.info(f"Final combined dataset: {final_file}")
        logger.info("=" * 60)
        return True
    else:
        logger.error("FAILED: Could not create combined dataset")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)