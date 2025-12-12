#!/usr/bin/env python3
"""
Test script to collect Real-Time Hourly LMPs data for first 3 days of January 2014
to verify the approach works before proceeding with the full year.
"""

import os
import subprocess
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_first_3_days():
    """Test collecting data for first 3 days of January 2014."""
    logger.info("Testing Real-Time Hourly LMPs collection for Jan 1-3, 2014")
    
    # Create output directory
    output_dir = "02_data/raw/pjm/real_time_hourly_lmps/test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file
    output_file = os.path.join(output_dir, "rt_hrl_lmps_jan1_3_2014_test.csv")
    
    try:
        # Build command to collect data
        cmd = (
            f'cd pjm_dataminer-master && set PYTHONIOENCODING=utf-8 && '
            f'venv\\Scripts\\activate && python fetch_pjm.py '
            f'-u rt_hrl_lmps -f csv -o ..\\{output_file}'
        )
        
        logger.info("Running command to collect RT Hourly LMPs data...")
        logger.info(f"Command: {cmd}")
        
        # Execute command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            # Check file size
            file_size = os.path.getsize(output_file)
            if file_size > 0:
                logger.info(f"SUCCESS: Data collected successfully!")
                logger.info(f"File: {output_file}")
                logger.info(f"File size: {file_size/1024/1024:.2f} MB")
                
                # Quick check of data
                try:
                    df = pd.read_csv(output_file)
                    logger.info(f"Total rows: {len(df)}")
                    logger.info(f"Columns: {list(df.columns)}")
                    
                    # Show first few rows
                    logger.info("First 3 rows:")
                    logger.info(df.head(3).to_string())
                    
                    # Check for 2014 data
                    date_columns = []
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                            date_columns.append(col)
                    
                    if date_columns:
                        date_col = date_columns[0]
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        min_date = df[date_col].min()
                        max_date = df[date_col].max()
                        logger.info(f"Date range: {min_date} to {max_date}")
                        
                        # Count 2014 data
                        data_2014 = df[df[date_col].dt.year == 2014]
                        logger.info(f"Rows from 2014: {len(data_2014)}")
                        
                        if len(data_2014) > 0:
                            logger.info("✅ SUCCESS: Found 2014 data in the results!")
                            return True
                        else:
                            logger.warning("⚠️  WARNING: No 2014 data found in the results")
                            return False
                    else:
                        logger.warning("No date column found")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error reading CSV: {str(e)}")
                    return False
                
            else:
                logger.error("FAILED: File is empty")
                return False
        else:
            logger.error("FAILED: Command failed or file not created")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("FAILED: Command timed out")
        return False
    except Exception as e:
        logger.error(f"FAILED: Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TEST: Real-Time Hourly LMPs for Jan 1-3, 2014")
    logger.info("=" * 60)
    
    success = test_first_3_days()
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ TEST PASSED: Ready to proceed with full year collection")
        logger.info("Next step: Run the full 122-chunk collection script")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("❌ TEST FAILED: Need to troubleshoot before proceeding")
        logger.info("=" * 60)