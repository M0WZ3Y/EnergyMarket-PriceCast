#!/usr/bin/env python3
"""
Process Downloaded 2014 LMP Data
Processes the currently downloading PJM data files and extracts 2014 data
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process downloaded PJM data files"""
    
    def __init__(self):
        self.pjm_dir = os.path.join(project_root, 'pjm_dataminer-master')
        self.data_dir = os.path.join(project_root, '02_data', 'raw', 'pjm')
        self.da_output_dir = os.path.join(self.data_dir, 'day_ahead_hourly_lmps')
        self.settlement_output_dir = os.path.join(self.data_dir, 'settlement_verified_hourly_lmps')
        
        # Create output directories
        os.makedirs(self.da_output_dir, exist_ok=True)
        os.makedirs(self.settlement_output_dir, exist_ok=True)
        
        logger.info("Data Processor initialized")
        logger.info(f"Project root: {project_root}")
        logger.info(f"PJM directory: {self.pjm_dir}")
        logger.info(f"Data output directory: {self.data_dir}")
    
    def process_day_ahead_data(self):
        """Process Day-Ahead Hourly LMPs data"""
        logger.info("=" * 60)
        logger.info("PROCESSING DAY-AHEAD HOURLY LMPs DATA")
        logger.info("=" * 60)
        
        # Check for downloaded file
        source_file = os.path.join(self.pjm_dir, 'da_hrl_lmps_2014.csv')
        
        if not os.path.exists(source_file):
            logger.warning(f"Day-Ahead LMP file not found: {source_file}")
            logger.info("Waiting for download to complete...")
            return False
        
        logger.info(f"Processing Day-Ahead LMP file: {source_file}")
        
        try:
            # Get file size
            file_size = os.path.getsize(source_file) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
            # Read in chunks to handle large file
            chunk_size = 50000
            chunks = []
            total_rows = 0
            
            logger.info("Reading data in chunks...")
            for chunk in pd.read_csv(source_file, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.info(f"Read {total_rows:,} rows...")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total records in file: {len(df):,}")
            
            # Check date column and filter for 2014
            if 'datetime_beginning_ept' in df.columns:
                df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
                df_2014 = df[df['datetime_beginning_ept'].dt.year == 2014]
                
                logger.info(f"Records from 2014: {len(df_2014):,}")
                
                # Save 2014 data
                output_2014 = os.path.join(self.da_output_dir, 'da_hrl_lmps_2014.csv')
                df_2014.to_csv(output_2014, index=False)
                logger.info(f"2014 Day-Ahead LMP data saved to: {output_2014}")
                
                # Save summary
                summary_file = os.path.join(self.da_output_dir, 'da_hrl_lmps_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write(f"Day-Ahead Hourly LMPs Data Summary\n")
                    f.write(f"=====================================\n")
                    f.write(f"Total records downloaded: {len(df):,}\n")
                    f.write(f"Records from 2014: {len(df_2014):,}\n")
                    f.write(f"Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}\n")
                    f.write(f"2014 date range: {df_2014['datetime_beginning_ept'].min()} to {df_2014['datetime_beginning_ept'].max()}\n")
                    f.write(f"\nColumns: {list(df.columns)}\n")
                
                logger.info(f"Summary saved to: {summary_file}")
                return True
                
            else:
                logger.warning("Date column not found in Day-Ahead data")
                return False
                
        except Exception as e:
            logger.error(f"Error processing Day-Ahead LMP data: {str(e)}")
            return False
    
    def process_settlement_data(self):
        """Process Settlements Verified Hourly LMPs data"""
        logger.info("=" * 60)
        logger.info("PROCESSING SETTLEMENTS VERIFIED HOURLY LMPs DATA")
        logger.info("=" * 60)
        
        # Check for downloaded file
        source_file = os.path.join(self.pjm_dir, 'test_settlement_lmps.csv')
        
        if not os.path.exists(source_file):
            logger.warning(f"Settlement LMP file not found: {source_file}")
            logger.info("Waiting for download to complete...")
            return False
        
        logger.info(f"Processing Settlement LMP file: {source_file}")
        
        try:
            # Get file size
            file_size = os.path.getsize(source_file) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
            # Read in chunks to handle large file
            chunk_size = 50000
            chunks = []
            total_rows = 0
            
            logger.info("Reading data in chunks...")
            for chunk in pd.read_csv(source_file, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.info(f"Read {total_rows:,} rows...")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Total records in file: {len(df):,}")
            
            # Check date column and filter for 2014
            if 'datetime_beginning_ept' in df.columns:
                df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
                df_2014 = df[df['datetime_beginning_ept'].dt.year == 2014]
                
                logger.info(f"Records from 2014: {len(df_2014):,}")
                
                # Analyze data types
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
                output_2014 = os.path.join(self.settlement_output_dir, 'settlement_lmps_2014.csv')
                df_2014.to_csv(output_2014, index=False)
                logger.info(f"2014 Settlement LMP data saved to: {output_2014}")
                
                # Save summary
                summary_file = os.path.join(self.settlement_output_dir, 'settlement_lmps_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write(f"Settlements Verified Hourly LMPs Data Summary\n")
                    f.write(f"============================================\n")
                    f.write(f"Total records downloaded: {len(df):,}\n")
                    f.write(f"Records from 2014: {len(df_2014):,}\n")
                    f.write(f"Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}\n")
                    f.write(f"2014 date range: {df_2014['datetime_beginning_ept'].min()} to {df_2014['datetime_beginning_ept'].max()}\n")
                    f.write(f"\nColumns: {list(df.columns)}\n")
                    f.write(f"\nMarket Run Types (Complete Dataset):\n")
                    for market_type, count in market_types.items():
                        f.write(f"  {market_type}: {count:,} records\n")
                    f.write(f"\nMarket Run Types (2014 Only):\n")
                    for market_type, count in market_types_2014.items():
                        f.write(f"  {market_type}: {count:,} records\n")
                
                logger.info(f"Summary saved to: {summary_file}")
                return True
                
            else:
                logger.warning("Date column not found in Settlement data")
                return False
                
        except Exception as e:
            logger.error(f"Error processing Settlement LMP data: {str(e)}")
            return False
    
    def create_final_summary(self):
        """Create final summary of all processed data"""
        logger.info("=" * 60)
        logger.info("CREATING FINAL SUMMARY")
        logger.info("=" * 60)
        
        summary_file = os.path.join(self.data_dir, '2014_LMP_DATA_PROCESSING_SUMMARY.md')
        
        with open(summary_file, 'w') as f:
            f.write(f"# 2014 PJM LMP Data Processing Summary\n\n")
            f.write(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Datasets Processed\n\n")
            
            # Day-Ahead LMPs
            da_file = os.path.join(self.da_output_dir, 'da_hrl_lmps_2014.csv')
            if os.path.exists(da_file):
                da_size = os.path.getsize(da_file) / (1024 * 1024)  # MB
                f.write(f"### 1. Day-Ahead Hourly LMPs\n")
                f.write(f"- **Description:** Key feature for price forecasting models\n")
                f.write(f"- **File:** `da_hrl_lmps_2014.csv`\n")
                f.write(f"- **Size:** {da_size:.2f} MB\n")
                f.write(f"- **Location:** `{da_file}`\n")
                f.write(f"- **Status:** ✅ Processed and filtered for 2014\n\n")
            
            # Settlement LMPs
            settlement_file = os.path.join(self.settlement_output_dir, 'settlement_lmps_2014.csv')
            if os.path.exists(settlement_file):
                settlement_size = os.path.getsize(settlement_file) / (1024 * 1024)  # MB
                f.write(f"### 2. Settlements Verified Hourly LMPs\n")
                f.write(f"- **Description:** Most accurate historical price data\n")
                f.write(f"- **File:** `settlement_lmps_2014.csv`\n")
                f.write(f"- **Size:** {settlement_size:.2f} MB\n")
                f.write(f"- **Location:** `{settlement_file}`\n")
                f.write(f"- **Status:** ✅ Processed and filtered for 2014\n\n")
            
            f.write(f"## Usage\n\n")
            f.write(f"Both datasets are ready for use in electricity price forecasting models.\n")
            f.write(f"The Settlements Verified Hourly LMPs dataset is recommended as the primary\n")
            f.write(f"data source due to its verified accuracy and inclusion of both Real-Time\n")
            f.write(f"and Day-Ahead LMPs.\n\n")
            
            f.write(f"## Next Steps\n\n")
            f.write(f"1. Data quality validation\n")
            f.write(f"2. Feature engineering\n")
            f.write(f"3. Model development\n")
            f.write(f"4. Performance evaluation\n\n")
        
        logger.info(f"Final summary saved to: {summary_file}")
    
    def run_processing(self):
        """Run complete data processing"""
        logger.info("Starting 2014 LMP data processing...")
        
        success_count = 0
        
        # Process Day-Ahead Hourly LMPs
        if self.process_day_ahead_data():
            success_count += 1
        
        # Process Settlements Verified Hourly LMPs
        if self.process_settlement_data():
            success_count += 1
        
        # Create final summary
        self.create_final_summary()
        
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully processed {success_count}/2 datasets")
        
        if success_count == 2:
            logger.info("All 2014 LMP data processing completed successfully!")
        else:
            logger.info("Some datasets may still be downloading...")
        
        return success_count

def main():
    """Main function"""
    processor = DataProcessor()
    success_count = processor.run_processing()
    
    if success_count > 0:
        logger.info("Data processing completed!")
        return 0
    else:
        logger.info("No data processed - downloads may still be in progress")
        return 1

if __name__ == "__main__":
    exit(main())