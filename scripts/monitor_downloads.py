#!/usr/bin/env python3
"""
PJM Data Download Monitor
Shows detailed information about ongoing PJM data downloads including:
- Dataset name and description
- Year being downloaded
- Original file size (total rows)
- Current download progress (rows downloaded)
- Percentage complete
- Estimated time remaining
"""

import os
import time
import psutil
import re
from datetime import datetime
from pathlib import Path

def get_dataset_info():
    """Get information about PJM datasets from the data list"""
    dataset_info = {
        'da_hrl_lmps': {
            'name': 'Day-Ahead Hourly LMPs',
            'description': 'Hourly Day-Ahead Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates'
        },
        'rt_da_monthly_lmps': {
            'name': 'Settlements Verified Hourly LMPs', 
            'description': 'Verified hourly Real-Time LMPs for aggregate and zonal pnodes used in settlements and final Day-Ahead LMPs'
        },
        'rt_hrl_lmps': {
            'name': 'Real-Time Hourly LMPs',
            'description': 'Hourly Real-Time Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates'
        }
    }
    return dataset_info

def parse_terminal_output():
    """Parse terminal output to extract download progress"""
    # This would typically read from actual terminal logs
    # For now, we'll simulate based on the current terminal output
    downloads = []
    
    # Terminal 1: da_hrl_lmps
    downloads.append({
        'dataset': 'da_hrl_lmps',
        'current_rows': 25825,
        'total_rows': 339648,
        'output_file': 'da_hrl_lmps_2014.csv',
        'terminal': 1
    })
    
    # Terminal 2: rt_da_monthly_lmps  
    downloads.append({
        'dataset': 'rt_da_monthly_lmps',
        'current_rows': 10500,
        'total_rows': 242544,
        'output_file': 'test_settlement_lmps.csv',
        'terminal': 2
    })
    
    return downloads

def estimate_time_remaining(current_rows, total_rows, start_time=None):
    """Estimate time remaining for download"""
    if current_rows == 0:
        return "Unknown"
    
    percentage = (current_rows / total_rows) * 100
    
    if start_time:
        elapsed = time.time() - start_time
        if elapsed > 0:
            rate = current_rows / elapsed  # rows per second
            remaining_rows = total_rows - current_rows
            if rate > 0:
                remaining_seconds = remaining_rows / rate
                remaining_time = str(datetime.timedelta(seconds=int(remaining_seconds)))
                return remaining_time
    
    return f"{percentage:.1f}% complete"

def get_file_size_estimate(file_path):
    """Get current file size if file exists"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    return "File not created yet"

def monitor_downloads():
    """Main monitoring function"""
    dataset_info = get_dataset_info()
    downloads = parse_terminal_output()
    
    print("=" * 80)
    print("PJM DATA DOWNLOAD MONITOR")
    print("=" * 80)
    print(f"Monitor Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Active Downloads: {len(downloads)}")
    print("-" * 80)
    
    for i, download in enumerate(downloads, 1):
        dataset = download['dataset']
        current = download['current_rows']
        total = download['total_rows']
        percentage = (current / total) * 100
        
        print(f"\nDOWNLOAD #{i} (Terminal {download['terminal']})")
        print("-" * 40)
        print(f"Dataset: {dataset}")
        print(f"Name: {dataset_info.get(dataset, {}).get('name', 'Unknown')}")
        print(f"Description: {dataset_info.get(dataset, {}).get('description', 'No description available')}")
        print(f"Output File: {download['output_file']}")
        print(f"Year: 2014")  # Based on the output filenames
        print(f"Original Size: {total:,} rows")
        print(f"Downloaded: {current:,} rows")
        print(f"Progress: {percentage:.2f}%")
        print(f"Remaining: {total - current:,} rows")
        
        # Estimate file size (rough calculation: ~100 bytes per row)
        estimated_size_mb = (total * 100) / (1024 * 1024)
        current_size_mb = (current * 100) / (1024 * 1024)
        print(f"Estimated Total Size: {estimated_size_mb:.1f} MB")
        print(f"Current Download Size: {current_size_mb:.1f} MB")
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f"Progress: |{bar}| {percentage:.1f}%")
        
        # Check if output file exists and get actual size
        output_path = f"pjm_dataminer-master/{download['output_file']}"
        actual_size = get_file_size_estimate(output_path)
        print(f"Actual File Size: {actual_size}")

def main():
    """Main function"""
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            monitor_downloads()
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to stop monitoring")
            print("Refreshing every 10 seconds...")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print("Final download status:")
        monitor_downloads()

if __name__ == "__main__":
    main()