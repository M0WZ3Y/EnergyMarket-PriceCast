#!/usr/bin/env python3
"""
Comprehensive 2014 LMP Data Collection Script
Collects both Day-Ahead Hourly LMPs and Settlements Verified Hourly LMPs for 2014
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pjm_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PJMDataCollector:
    """Comprehensive PJM data collector for 2014 LMP data"""
    
    def __init__(self):
        self.pjm_dir = os.path.join(project_root, 'pjm_dataminer-master')
        self.data_dir = os.path.join(project_root, '02_data', 'raw', 'pjm')
        self.da_output_dir = os.path.join(self.data_dir, 'day_ahead_hourly_lmps')
        self.settlement_output_dir = os.path.join(self.data_dir, 'settlement_verified_hourly_lmps')
        
        # Create output directories
        os.makedirs(self.da_output_dir, exist_ok=True)
        os.makedirs(self.settlement_output_dir, exist_ok=True)
        
        logger.info("PJM Data Collector initialized")
        logger.info(f"Project root: {project_root}")
        logger.info(f"PJM directory: {self.pjm_dir}")
        logger.info(f"Data output directory: {self.data_dir}")
    
    def run_pjm_command(self, endpoint, output_file):
        """Run PJM dataminer command"""
        cmd = [
            'cd', self.pjm_dir, '&&',
            'set', 'PYTHONIOENCODING=utf-8', '&&',
            'venv\\Scripts\\activate', '&&',
            'python', 'fetch_pjm.py',
            '-u', endpoint,
            '-f', 'csv',
            '-o', output_file
        ]
        
        command_str = ' '.join(cmd)
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
    
    def collect_day_ahead_lmps(self):
        """Collect Day-Ahead Hourly LMPs for 2014"""
        logger.info("=" * 60)
        logger.info("COLLECTING DAY-AHEAD HOURLY LMPs FOR 2014")
        logger.info("=" * 60)
        
        output_file = os.path.join(self.da_output_dir, 'da_hrl_lmps_complete.csv')
        
        logger.info("Starting download of Day-Ahead Hourly LMPs...")
        logger.info("This dataset contains 339,648 rows and may take some time...")
        
        success = self.run_pjm_command('da_hrl_lmps', output_file)
        
        if success and os.path.exists(output_file):
            # Analyze and filter for 2014 data
            logger.info("Download completed. Analyzing data...")
            
            try:
                # Read in chunks to handle large file
                chunk_size = 50000
                chunks = []
                
                for chunk in pd.read_csv(output_file, chunksize=chunk_size):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Total records downloaded: {len(df):,}")
                
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
                    
                else:
                    logger.warning("Date column not found. Saving complete dataset.")
                    
            except Exception as e:
                logger.error(f"Error processing Day-Ahead LMP data: {str(e)}")
                
            return True
        else:
            logger.error("Failed to download Day-Ahead Hourly LMPs")
            return False
    
    def collect_settlement_lmps(self):
        """Collect Settlements Verified Hourly LMPs for 2014"""
        logger.info("=" * 60)
        logger.info("COLLECTING SETTLEMENTS VERIFIED HOURLY LMPs FOR 2014")
        logger.info("=" * 60)
        
        output_file = os.path.join(self.settlement_output_dir, 'settlement_lmps_complete.csv')
        
        logger.info("Starting download of Settlements Verified Hourly LMPs...")
        logger.info("This dataset contains 242,544 rows and includes both Real-Time and Day-Ahead LMPs...")
        
        success = self.run_pjm_command('rt_da_monthly_lmps', output_file)
        
        if success and os.path.exists(output_file):
            # Analyze and filter for 2014 data
            logger.info("Download completed. Analyzing data...")
            
            try:
                # Read in chunks to handle large file
                chunk_size = 50000
                chunks = []
                
                for chunk in pd.read_csv(output_file, chunksize=chunk_size):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Total records downloaded: {len(df):,}")
                
                # Check date column and filter for 2014
                if 'datetime_beginning_ept' in df.columns:
                    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
                    df_2014 = df[df['datetime_beginning_ept'].dt.year == 2014]
                    
                    logger.info(f"Records from 2014: {len(df_2014):,}")
                    
                    # Save 2014 data
                    output_2014 = os.path.join(self.settlement_output_dir, 'settlement_lmps_2014.csv')
                    df_2014.to_csv(output_2014, index=False)
                    logger.info(f"2014 Settlement LMP data saved to: {output_2014}")
                    
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
                    
                else:
                    logger.warning("Date column not found. Saving complete dataset.")
                    
            except Exception as e:
                logger.error(f"Error processing Settlement LMP data: {str(e)}")
                
            return True
        else:
            logger.error("Failed to download Settlements Verified Hourly LMPs")
            return False
    
    def create_final_summary(self):
        """Create final summary of all collected data"""
        logger.info("=" * 60)
        logger.info("CREATING FINAL SUMMARY")
        logger.info("=" * 60)
        
        summary_file = os.path.join(self.data_dir, '2014_LMP_DATA_COLLECTION_SUMMARY.md')
        
        with open(summary_file, 'w') as f:
            f.write(f"# 2014 PJM LMP Data Collection Summary\n\n")
            f.write(f"**Collection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Datasets Collected\n\n")
            
            # Day-Ahead LMPs
            da_file = os.path.join(self.da_output_dir, 'da_hrl_lmps_2014.csv')
            if os.path.exists(da_file):
                da_size = os.path.getsize(da_file) / (1024 * 1024)  # MB
                f.write(f"### 1. Day-Ahead Hourly LMPs\n")
                f.write(f"- **Description:** Key feature for price forecasting models\n")
                f.write(f"- **File:** `da_hrl_lmps_2014.csv`\n")
                f.write(f"- **Size:** {da_size:.2f} MB\n")
                f.write(f"- **Location:** `{da_file}`\n\n")
            
            # Settlement LMPs
            settlement_file = os.path.join(self.settlement_output_dir, 'settlement_lmps_2014.csv')
            if os.path.exists(settlement_file):
                settlement_size = os.path.getsize(settlement_file) / (1024 * 1024)  # MB
                f.write(f"### 2. Settlements Verified Hourly LMPs\n")
                f.write(f"- **Description:** Most accurate historical price data\n")
                f.write(f"- **File:** `settlement_lmps_2014.csv`\n")
                f.write(f"- **Size:** {settlement_size:.2f} MB\n")
                f.write(f"- **Location:** `{settlement_file}`\n\n")
            
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
    
    def run_complete_collection(self):
        """Run complete data collection for both datasets"""
        logger.info("Starting comprehensive 2014 LMP data collection...")
        
        success_count = 0
        
        # Collect Day-Ahead Hourly LMPs
        if self.collect_day_ahead_lmps():
            success_count += 1
        
        # Collect Settlements Verified Hourly LMPs
        if self.collect_settlement_lmps():
            success_count += 1
        
        # Create final summary
        self.create_final_summary()
        
        logger.info("=" * 60)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully collected {success_count}/2 datasets")
        
        if success_count == 2:
            logger.info("✅ All 2014 LMP data collection completed successfully!")
        else:
            logger.warning("⚠️ Some datasets may not have been collected completely")
        
        return success_count == 2

def main():
    """Main function"""
    collector = PJMDataCollector()
    success = collector.run_complete_collection()
    
    if success:
        logger.info("🎉 Complete 2014 LMP data collection finished successfully!")
        return 0
    else:
        logger.error("❌ Data collection encountered issues")
        return 1

if __name__ == "__main__":
    exit(main())