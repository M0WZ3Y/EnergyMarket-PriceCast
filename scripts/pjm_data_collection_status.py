#!/usr/bin/env python3
"""
PJM Data Collection Status and Time Estimation
Provides comprehensive status of current downloads and estimates remaining time
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def get_current_download_status():
    """Get current download status from monitoring data"""
    
    # Current active downloads based on terminal monitoring
    current_downloads = {
        'da_hrl_lmps_2014': {
            'dataset': 'da_hrl_lmps',
            'name': 'Day-Ahead Hourly LMPs',
            'year': 2014,
            'total_rows': 339648,
            'downloaded_rows': 27625,  # Based on latest monitor output
            'progress_percent': 8.13,
            'speed_rows_per_sec': 46.0,
            'estimated_size_mb': 32.4
        },
        'rt_da_monthly_lmps_2014': {
            'dataset': 'rt_da_monthly_lmps',
            'name': 'Settlements Verified Hourly LMPs',
            'year': 2014,
            'total_rows': 242544,
            'downloaded_rows': 12325,  # Based on latest monitor output
            'progress_percent': 5.08,
            'speed_rows_per_sec': 20.5,
            'estimated_size_mb': 23.1
        }
    }
    
    # Multi-year collection plan
    years_to_collect = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    datasets_to_collect = ['da_hrl_lmps', 'rt_da_monthly_lmps']
    
    return {
        'current_downloads': current_downloads,
        'years_to_collect': years_to_collect,
        'datasets_to_collect': datasets_to_collect
    }

def calculate_time_estimates(status):
    """Calculate time estimates for completing all downloads"""
    
    current_downloads = status['current_downloads']
    years = status['years_to_collect']
    datasets = status['datasets_to_collect']
    
    # Calculate remaining time for current 2014 downloads
    total_remaining_rows_2014 = 0
    combined_speed_2014 = 0
    
    for download_id, download_info in current_downloads.items():
        remaining_rows = download_info['total_rows'] - download_info['downloaded_rows']
        total_remaining_rows_2014 += remaining_rows
        combined_speed_2014 += download_info['speed_rows_per_sec']
    
    # Time to complete 2014 downloads
    time_to_complete_2014_hours = total_remaining_rows_2014 / combined_speed_2014 / 3600 if combined_speed_2014 > 0 else float('inf')
    
    # Estimate data size for remaining years (assuming similar size to 2014)
    estimated_rows_per_year = {
        'da_hrl_lmps': 339648,  # Based on 2014 data
        'rt_da_monthly_lmps': 242544  # Based on 2014 data
    }
    
    # Calculate total data for all years
    total_rows_all_years = 0
    for year in years:
        for dataset in datasets:
            if year == 2014:
                # 2014 is already being downloaded
                total_rows_all_years += estimated_rows_per_year[dataset]
            else:
                # Future years
                total_rows_all_years += estimated_rows_per_year[dataset]
    
    # Calculate total time for all downloads
    # Assuming same average speed as current downloads
    avg_speed = combined_speed_2014 / len(current_downloads) if current_downloads else 0
    total_time_all_years_hours = total_rows_all_years / avg_speed / 3600 if avg_speed > 0 else float('inf')
    
    # Calculate time for remaining years (2015-2024)
    remaining_years = [year for year in years if year != 2014]
    total_rows_remaining_years = 0
    for year in remaining_years:
        for dataset in datasets:
            total_rows_remaining_years += estimated_rows_per_year[dataset]
    
    time_remaining_years_hours = total_rows_remaining_years / avg_speed / 3600 if avg_speed > 0 else float('inf')
    
    return {
        'time_to_complete_2014_hours': time_to_complete_2014_hours,
        'time_remaining_years_hours': time_remaining_years_hours,
        'total_time_all_years_hours': total_time_all_years_hours,
        'total_rows_all_years': total_rows_all_years,
        'total_rows_remaining_years': total_rows_remaining_years,
        'avg_speed_rows_per_sec': avg_speed
    }

def format_time(hours):
    """Format hours into human-readable format"""
    if hours == float('inf'):
        return "Unknown (speed too low to calculate)"
    
    days = int(hours // 24)
    remaining_hours = int(hours % 24)
    minutes = int((hours % 1) * 60)
    
    if days > 0:
        return f"{days} days, {remaining_hours} hours, {minutes} minutes"
    elif hours > 1:
        return f"{remaining_hours} hours, {minutes} minutes"
    else:
        return f"{int(hours * 60)} minutes"

def generate_status_report():
    """Generate comprehensive status report"""
    
    print("=" * 80)
    print("PJM DATA COLLECTION STATUS AND TIME ESTIMATION")
    print("=" * 80)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get current status
    status = get_current_download_status()
    time_estimates = calculate_time_estimates(status)
    
    # Current downloads status
    print("CURRENT DOWNLOAD STATUS")
    print("-" * 40)
    for download_id, download_info in status['current_downloads'].items():
        print(f"Dataset: {download_info['name']}")
        print(f"Year: {download_info['year']}")
        print(f"Progress: {download_info['downloaded_rows']:,} / {download_info['total_rows']:,} rows ({download_info['progress_percent']:.2f}%)")
        print(f"Speed: {download_info['speed_rows_per_sec']:.1f} rows/sec")
        print(f"Estimated size: {download_info['estimated_size_mb']:.1f} MB")
        remaining_rows = download_info['total_rows'] - download_info['downloaded_rows']
        remaining_time = remaining_rows / download_info['speed_rows_per_sec'] / 3600 if download_info['speed_rows_per_sec'] > 0 else float('inf')
        print(f"Time remaining: {format_time(remaining_time)}")
        print()
    
    # Overall progress for 2014
    total_2014_rows = sum(d['total_rows'] for d in status['current_downloads'].values())
    total_downloaded_2014 = sum(d['downloaded_rows'] for d in status['current_downloads'].values())
    progress_2014 = (total_downloaded_2014 / total_2014_rows * 100) if total_2014_rows > 0 else 0
    
    print("2014 DATA COLLECTION SUMMARY")
    print("-" * 40)
    print(f"Total rows to download: {total_2014_rows:,}")
    print(f"Rows downloaded: {total_downloaded_2014:,}")
    print(f"Overall progress: {progress_2014:.2f}%")
    print(f"Combined speed: {time_estimates['avg_speed_rows_per_sec']:.1f} rows/sec")
    print(f"Time to complete 2014: {format_time(time_estimates['time_to_complete_2014_hours'])}")
    print()
    
    # Multi-year collection plan
    print("MULTI-YEAR COLLECTION PLAN")
    print("-" * 40)
    print(f"Years to collect: {', '.join(map(str, status['years_to_collect']))}")
    print(f"Datasets: {', '.join(status['datasets_to_collect'])}")
    print(f"Total years: {len(status['years_to_collect'])}")
    print(f"Total datasets: {len(status['datasets_to_collect'])}")
    print(f"Total dataset-years: {len(status['years_to_collect']) * len(status['datasets_to_collect'])}")
    print()
    
    # Time estimates for complete collection
    print("TIME ESTIMATES FOR COMPLETE COLLECTION")
    print("-" * 40)
    print(f"Total rows for all years: {time_estimates['total_rows_all_years']:,}")
    print(f"Rows for remaining years (2015-2024): {time_estimates['total_rows_remaining_years']:,}")
    print(f"Average download speed: {time_estimates['avg_speed_rows_per_sec']:.1f} rows/sec")
    print()
    print(f"Time to complete 2014 data: {format_time(time_estimates['time_to_complete_2014_hours'])}")
    print(f"Time for remaining years: {format_time(time_estimates['time_remaining_years_hours'])}")
    print(f"Total time for all years: {format_time(time_estimates['total_time_all_years_hours'])}")
    print()
    
    # Projected completion dates
    now = datetime.now()
    completion_2014 = now + timedelta(hours=time_estimates['time_to_complete_2014_hours'])
    completion_all = now + timedelta(hours=time_estimates['total_time_all_years_hours'])
    
    print("PROJECTED COMPLETION DATES")
    print("-" * 40)
    print(f"2014 data completion: {completion_2014.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All years completion: {completion_all.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS")
    print("-" * 40)
    print("1. Continue monitoring current 2014 downloads")
    print("2. Once 2014 is complete, process and validate the data")
    print("3. Start multi-year collection for 2015-2024 using the fixed script")
    print("4. Consider running downloads during off-peak hours for better speeds")
    print("5. Monitor disk space - total data will be approximately:")
    
    total_size_mb = sum(d['estimated_size_mb'] for d in status['current_downloads'].values()) * len(status['years_to_collect'])
    print(f"   - Estimated {total_size_mb:.1f} MB for all years")
    print(f"   - Approximately {total_size_mb / 1024:.2f} GB total")
    print()
    
    print("=" * 80)
    print("END OF STATUS REPORT")
    print("=" * 80)

if __name__ == "__main__":
    generate_status_report()