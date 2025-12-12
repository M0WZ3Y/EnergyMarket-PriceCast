#!/usr/bin/env python3
"""
Process Current Downloads
Processes the currently downloading 2014 data files once they complete
"""

import os
import sys
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
        logging.FileHandler('process_current_downloads.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CurrentDownloadProcessor:
    """Process currently downloading PJM data files"""
    
    def __init__(self):
        self.pjm_dir = os.path.join(project_root, 'pjm_dataminer-master')
        self.data_dir = os.path.join(project_root, '02_data', 'raw', 'pjm')
        
        # Current download files
        self.downloads = {
            'da_hrl_lmps': {
                'name': 'Day-Ahead Hourly LMPs',
                'source_file': os.path.join(self.pjm_dir, 'da_hrl_lmps_2014.csv'),
                'output_dir': os.path.join(self.data_dir, 'day_ahead_hourly_lmps'),
                'output_file': os.path.join(self.data_dir, 'day_ahead_hourly_lmps', 'da_hrl_lmps_2014.csv')
            },
            'rt_da_monthly_lmps': {
                'name': 'Settlements Verified Hourly LMPs',
                'source_file': os.path.join(self.pjm_dir, 'test_settlement_lmps.csv'),
                'output_dir': os.path.join(self.data_dir, 'settlement_verified_hourly_lmps'),
                'output_file': os.path.join(self.data_dir, 'settlement_verified_hourly_lmps', 'settlement_lmps_2014.csv')
            }
        }
        
        # Create output directories
        for download_info in self.downloads.values():
            os.makedirs(download_info['output_dir'], exist_ok=True)
        
        logger.info("Current Download Processor initialized")
        logger.info(f"Project root: {project_root}")
        logger.info(f"PJM directory: {self.pjm_dir}")
        logger.info(f"Data output directory: {self.data_dir}")
    
    def check_download_complete(self, file_path):
        """Check if download is complete by monitoring file size stability"""
        if not os.path.exists(file_path):
            return False, "File not found"
        
        # Check file size stability (no growth for 30 seconds)
        sizes = []
        for i in range(6):  # Check for 30 seconds (6 * 5 seconds)
            try:
                size = os.path.getsize(file_path)
                sizes.append(size)
                logger.info(f"Check {i+1}: File size = {size:,} bytes")
            except:
                return False, "Error accessing file"
            
            if i < 5:  # Don't sleep after last check
                time.sleep(5)
        
        # Check if file size is stable
        if len(set(sizes[-3:])) == 1:  # Last 3 checks have same size
            return True, f"Download complete (stable size: {sizes[-1]:,} bytes)"
        else:
            return False, f"Download still in progress (size changing: {sizes})"
    
    def process_dataset(self, dataset_key):
        """Process a specific dataset"""
        dataset_info = self.downloads[dataset_key]
        
        logger.info("=" * 80)
        logger.info(f"PROCESSING {dataset_info['name'].upper()}")
        logger.info("=" * 80)
        
        # Check if download is complete
        is_complete, status = self.check_download_complete(dataset_info['source_file'])
        logger.info(f"Download status: {status}")
        
        if not is_complete:
            logger.warning(f"Download not complete for {dataset_info['name']}")
            return False
        
        logger.info(f"Processing {dataset_info['name']}...")
        
        try:
            # Get file size
            file_size = os.path.getsize(dataset_info['source_file']) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
            # Read in chunks to handle large file
            chunk_size = 50000
            chunks = []
            total_rows = 0
            
            logger.info("Reading data in chunks...")
            for chunk in pd.read_csv(dataset_info['source_file'], chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows % 10000 == 0:
                    logger.info(f"Read {total_rows:,} rows...")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total records in file: {len(df):,}")
            
            # Check date column and filter for 2014
            if 'datetime_beginning_ept' in df.columns:
                df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
                df_2014 = df[df['datetime_beginning_ept'].dt.year == 2014]
                
                logger.info(f"Records from 2014: {len(df_2014):,}")
                
                # Analyze data types if available
                if 'market_run_type' in df.columns:
                    market_types = df['market_run_type'].value_counts()
                    logger.info("Market run types in complete dataset:")
                    for market_type, count in market_types.items():
                        logger.info(f"  {market_type}: {count:,} records")
                    
                    # 2014 market types
                    market_types_2014 = df_2014['market_run_type'].value_counts()
                    logger.info("Market run types in 2014 dataset:")
                    for market_type, count in market_types_2014.items():
                        logger.info(f"  {market_type}: {count:,} records")
                
                # Save 2014 data
                df_2014.to_csv(dataset_info['output_file'], index=False)
                logger.info(f"2014 {dataset_info['name']} data saved to: {dataset_info['output_file']}")
                
                # Save summary
                summary_file = os.path.join(dataset_info['output_dir'], f'{dataset_key}_2014_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write(f"{dataset_info['name']} Data Summary - 2014\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Total records downloaded: {len(df):,}\n")
                    f.write(f"Records from 2014: {len(df_2014):,}\n")
                    f.write(f"Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}\n")
                    f.write(f"2014 date range: {df_2014['datetime_beginning_ept'].min()} to {df_2014['datetime_beginning_ept'].max()}\n")
                    f.write(f"\nColumns: {list(df.columns)}\n")
                    
                    if 'market_run_type' in df.columns:
                        f.write(f"\nMarket Run Types (Complete Dataset):\n")
                        for market_type, count in market_types.items():
                            f.write(f"  {market_type}: {count:,} records\n")
                        f.write(f"\nMarket Run Types (2014 Only):\n")
                        for market_type, count in market_types_2014.items():
                            f.write(f"  {market_type}: {count:,} records\n")
                
                logger.info(f"Summary saved to: {summary_file}")
                return True
                
            else:
                logger.warning("Date column not found in data")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {dataset_info['name']}: {str(e)}")
            return False
    
    def run_processing(self):
        """Run processing for all datasets"""
        logger.info("Starting processing of current downloads...")
        
        success_count = 0
        
        for dataset_key in self.downloads.keys():
            logger.info(f"Processing dataset: {dataset_key}")
            if self.process_dataset(dataset_key):
                success_count += 1
                logger.info(f"✅ Successfully processed {self.downloads[dataset_key]['name']}")
            else:
                logger.error(f"❌ Failed to process {self.downloads[dataset_key]['name']}")
        
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Successfully processed {success_count}/{len(self.downloads)} datasets")
        
        if success_count == len(self.downloads):
            logger.info("🎉 All current downloads processed successfully!")
            logger.info("You can now run the multi-year collection script:")
            logger.info("python scripts/collect_multi_year_lmp_data.py")
        else:
            logger.warning("⚠️ Some datasets may still be downloading")
        
        return success_count == len(self.downloads)

def main():
    """Main function"""
    processor = CurrentDownloadProcessor()
    success = processor.run_processing()
    
    if success:
        logger.info("🎉 Current download processing completed!")
        return 0
    else:
        logger.info("⏳ Downloads may still be in progress")
        return 1

if __name__ == "__main__":
    exit(main())