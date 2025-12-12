#!/usr/bin/env python3
"""
Ultimate PJM Download Monitor - Combines the best features of all existing monitors
Provides real-time monitoring with accurate progress tracking and time estimation
"""

import os
import re
import time
import sys
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psutil

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_RED = '\033[101m'
    BG_BLUE = '\033[104m'
    BG_CYAN = '\033[106m'

class UltimatePJMMonitor:
    """Ultimate PJM Download Monitor with real-time progress tracking"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.last_progress = {}
        self.progress_history = {}
        self.speed_calculators = {}
        
    def get_dataset_info(self):
        """Get information about PJM datasets"""
        return {
            'da_hrl_lmps': {
                'name': 'Day-Ahead Hourly LMPs',
                'description': 'Hourly Day-Ahead Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates',
                'total_rows': 339648,
                'output_file': 'da_hrl_lmps_2014.csv'
            },
            'rt_da_monthly_lmps': {
                'name': 'Settlements Verified Hourly LMPs',
                'description': 'Verified hourly Real-Time LMPs for aggregate and zonal pnodes used in settlements and final Day-Ahead LMPs',
                'total_rows': 242544,
                'output_file': 'test_settlement_lmps.csv'
            }
        }
    
    def parse_terminal_output_realtime(self):
        """Parse actual terminal output to get current download progress"""
        progress_data = {}
        
        # Method 1: Parse from running processes
        try:
            # Get all python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'fetch_pjm.py' in cmdline:
                        # Determine dataset type
                        if '-u da_hrl_lmps' in cmdline:
                            dataset = 'da_hrl_lmps'
                        elif '-u rt_da_monthly_lmps' in cmdline:
                            dataset = 'rt_da_monthly_lmps'
                        else:
                            continue
                        
                        # Try to get the latest output from the process
                        try:
                            # This is a simplified approach - in practice you'd need to capture stdout
                            # For now, we'll use file-based progress tracking
                            output_file = self.get_dataset_info()[dataset]['output_file']
                            if os.path.exists(f"pjm_dataminer-master/{output_file}"):
                                # Count lines in the file
                                result = subprocess.run(
                                    ['powershell', '-Command', 
                                     f'Get-Content "pjm_dataminer-master/{output_file}" | Measure-Object -Line | Select-Object -ExpandProperty Lines'],
                                    capture_output=True, text=True, timeout=5
                                )
                                if result.stdout.strip():
                                    lines = int(result.stdout.strip()) - 1  # Subtract header
                                    progress_data[dataset] = {
                                        'current_rows': min(lines, self.get_dataset_info()[dataset]['total_rows']),
                                        'total_rows': self.get_dataset_info()[dataset]['total_rows'],
                                        'output_file': output_file,
                                        'pid': proc.info['pid']
                                    }
                        except:
                            pass
        except:
            pass
        
        # Method 2: Parse from terminal output patterns (fallback)
        if not progress_data:
            # Look for patterns in the current terminal output
            # This would need to be adapted based on your actual terminal output
            pass
        
        return progress_data
    
    def calculate_speed(self, dataset: str, current_rows: int):
        """Calculate download speed for a dataset"""
        now = datetime.now()
        
        if dataset not in self.progress_history:
            self.progress_history[dataset] = []
        
        # Add current progress to history
        self.progress_history[dataset].append({
            'timestamp': now,
            'rows': current_rows
        })
        
        # Keep only last 10 entries
        if len(self.progress_history[dataset]) > 10:
            self.progress_history[dataset].pop(0)
        
        # Calculate speed based on recent progress
        if len(self.progress_history[dataset]) >= 2:
            recent = self.progress_history[dataset][-2:]
            time_diff = (recent[1]['timestamp'] - recent[0]['timestamp']).total_seconds()
            row_diff = recent[1]['rows'] - recent[0]['rows']
            
            if time_diff > 0 and row_diff > 0:
                speed = row_diff / time_diff
                return speed
        
        return 0.0
    
    def format_bytes(self, bytes_size: float) -> str:
        """Format bytes into human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration into human readable format"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_speed_color(self, speed: float) -> str:
        """Get color based on download speed"""
        if speed == 0:
            return Colors.RED
        elif speed < 10:
            return Colors.YELLOW
        elif speed < 30:
            return Colors.CYAN
        else:
            return Colors.GREEN
    
    def get_progress_color(self, completion: float) -> str:
        """Get color based on completion percentage"""
        if completion < 10:
            return Colors.RED
        elif completion < 30:
            return Colors.YELLOW
        elif completion < 70:
            return Colors.CYAN
        else:
            return Colors.GREEN
    
    def display_download_status(self, dataset: str, data: Dict, speed: float):
        """Display status for a single download"""
        current_rows = data['current_rows']
        total_rows = data['total_rows']
        completion = (current_rows / total_rows * 100) if total_rows > 0 else 0
        remaining = total_rows - current_rows
        
        # Size calculations
        bytes_per_row = 100
        total_size = total_rows * bytes_per_row
        current_size = current_rows * bytes_per_row
        
        # Get actual file size
        actual_size = 0
        try:
            if os.path.exists(f"pjm_dataminer-master/{data['output_file']}"):
                actual_size = os.path.getsize(f"pjm_dataminer-master/{data['output_file']}")
        except:
            pass
        
        # ETA calculation
        if speed > 0:
            eta_seconds = remaining / speed
            eta = self.format_duration(eta_seconds)
        else:
            eta = "Unknown"
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * completion / 100)
        bar = '=' * filled + '-' * (bar_width - filled)
        progress_color = self.get_progress_color(completion)
        speed_color = self.get_speed_color(speed)
        
        dataset_info = self.get_dataset_info()[dataset]
        
        print(f"\n{Colors.BOLD}DOWNLOAD - {dataset_info['name']}{Colors.RESET}")
        print(f"{Colors.CYAN}{'-' * 60}{Colors.RESET}")
        print(f"{Colors.WHITE}Dataset:{Colors.RESET} {Colors.MAGENTA}{dataset}{Colors.RESET}")
        print(f"{Colors.WHITE}Description:{Colors.RESET} {dataset_info['description']}")
        print(f"{Colors.WHITE}Output File:{Colors.RESET} {data['output_file']}")
        print(f"{Colors.WHITE}Process ID:{Colors.RESET} {data.get('pid', 'Unknown')}")
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}>>> PROGRESS <<<{Colors.RESET}")
        print(f"{Colors.WHITE}Downloaded:{Colors.RESET} {current_rows:,} / {total_rows:,} rows")
        print(f"{Colors.WHITE}Completion:{Colors.RESET} {progress_color}{completion:.2f}%{Colors.RESET}")
        print(f"{Colors.WHITE}Remaining:{Colors.RESET} {remaining:,} rows")
        print(f"{Colors.WHITE}Progress:{Colors.RESET} [{progress_color}{bar}{Colors.RESET}] {progress_color}{completion:.1f}%{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}>>> SPEED & TIME <<<{Colors.RESET}")
        print(f"{Colors.WHITE}Current Speed:{Colors.RESET} {speed_color}{speed:.1f} rows/sec{Colors.RESET}")
        print(f"{Colors.WHITE}Estimated Time Remaining:{Colors.RESET} {speed_color}{eta}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}>>> FILE SIZE <<<{Colors.RESET}")
        print(f"{Colors.WHITE}Estimated Total:{Colors.RESET} {self.format_bytes(total_size)}")
        print(f"{Colors.WHITE}Current Download:{Colors.RESET} {self.format_bytes(current_size)}")
        print(f"{Colors.WHITE}Actual File Size:{Colors.RESET} {self.format_bytes(actual_size) if actual_size > 0 else 'File not created yet'}")
    
    def display_overall_summary(self, progress_data: Dict):
        """Display overall download summary"""
        if not progress_data:
            print(f"\n{Colors.YELLOW}No active downloads detected{Colors.RESET}")
            return
        
        total_downloaded = sum(data['current_rows'] for data in progress_data.values())
        total_rows = sum(data['total_rows'] for data in progress_data.values())
        overall_completion = (total_downloaded / total_rows * 100) if total_rows > 0 else 0
        
        total_size = total_rows * 100
        downloaded_size = total_downloaded * 100
        
        # Combined speed
        combined_speed = sum(self.speed_calculators.get(dataset, 0) for dataset in progress_data.keys())
        
        # Overall progress bar
        bar_width = 50
        filled = int(bar_width * overall_completion / 100)
        bar = '=' * filled + '-' * (bar_width - filled)
        
        print(f"\n{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE}>>> OVERALL SUMMARY <<<{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.WHITE}Active Downloads:{Colors.RESET} {len(progress_data)}")
        print(f"{Colors.WHITE}Total Rows Downloaded:{Colors.RESET} {Colors.GREEN}{total_downloaded:,} / {total_rows:,}{Colors.RESET}")
        print(f"{Colors.WHITE}Overall Progress:{Colors.RESET} {self.get_progress_color(overall_completion)}{overall_completion:.2f}%{Colors.RESET}")
        print(f"{Colors.WHITE}Total Data Size:{Colors.RESET} {self.format_bytes(downloaded_size)} / {self.format_bytes(total_size)}")
        print(f"{Colors.WHITE}Combined Speed:{Colors.RESET} {self.get_speed_color(combined_speed)}{combined_speed:.1f} rows/sec{Colors.RESET}")
        print(f"{Colors.WHITE}Overall Progress:{Colors.RESET} [{self.get_progress_color(overall_completion)}{bar}{Colors.RESET}] {self.get_progress_color(overall_completion)}{overall_completion:.1f}%{Colors.RESET}")
        
        if combined_speed > 0:
            remaining_rows = total_rows - total_downloaded
            eta_seconds = remaining_rows / combined_speed
            eta = self.format_duration(eta_seconds)
            completion_time = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"{Colors.WHITE}Combined ETA:{Colors.RESET} {self.get_speed_color(combined_speed)}{eta}{Colors.RESET}")
            print(f"{Colors.WHITE}Estimated Completion:{Colors.RESET} {Colors.CYAN}{completion_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    
    def display_multi_year_projection(self):
        """Display projection for multi-year collection"""
        dataset_info = self.get_dataset_info()
        
        # Calculate total data for all years (2014-2024)
        years = list(range(2014, 2025))
        total_datasets = len(years) * len(dataset_info)
        
        total_rows_all_years = 0
        for year in years:
            for dataset in dataset_info.values():
                total_rows_all_years += dataset['total_rows']
        
        print(f"\n{Colors.BOLD}{Colors.BG_CYAN}{Colors.WHITE}>>> MULTI-YEAR PROJECTION (2014-2024) <<<{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.WHITE}Years to Collect:{Colors.RESET} {len(years)} (2014-2024)")
        print(f"{Colors.WHITE}Datasets per Year:{Colors.RESET} {len(dataset_info)}")
        print(f"{Colors.WHITE}Total Dataset-Years:{Colors.RESET} {total_datasets}")
        print(f"{Colors.WHITE}Total Rows to Download:{Colors.RESET} {Colors.YELLOW}{total_rows_all_years:,}{Colors.RESET}")
        print(f"{Colors.WHITE}Estimated Total Size:{Colors.RESET} {Colors.YELLOW}{self.format_bytes(total_rows_all_years * 100)}{Colors.RESET}")
        
        # Get current combined speed
        current_speed = sum(self.speed_calculators.values())
        if current_speed > 0:
            eta_seconds = total_rows_all_years / current_speed
            eta = self.format_duration(eta_seconds)
            completion_time = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"{Colors.WHITE}Estimated Total Time:{Colors.RESET} {Colors.YELLOW}{eta}{Colors.RESET}")
            print(f"{Colors.WHITE}Projected Completion:{Colors.RESET} {Colors.YELLOW}{completion_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        else:
            print(f"{Colors.WHITE}Estimated Total Time:{Colors.RESET} {Colors.RED}Unknown (waiting for speed data){Colors.RESET}")
    
    def run(self):
        """Main monitoring loop"""
        print(f"{Colors.BOLD}{Colors.BG_CYAN}{Colors.WHITE}>>> ULTIMATE PJM DOWNLOAD MONITOR <<<{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.WHITE}Monitor Started:{Colors.RESET} {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.WHITE}Refresh Interval:{Colors.RESET} 15 seconds")
        print(f"{Colors.WHITE}Monitoring Features:{Colors.RESET} Real-time progress, Speed calculation, ETA projection")
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        
        try:
            while True:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Header
                current_time = datetime.now()
                print(f"{Colors.BOLD}{Colors.BG_CYAN}{Colors.WHITE}>>> ULTIMATE PJM DOWNLOAD MONITOR <<<{Colors.RESET}")
                print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
                print(f"{Colors.WHITE}Status Time:{Colors.RESET} {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{Colors.WHITE}Monitor Runtime:{Colors.RESET} {self.format_duration((current_time - self.start_time).total_seconds())}")
                
                # Get current progress
                progress_data = self.parse_terminal_output_realtime()
                
                # Calculate speeds
                for dataset, data in progress_data.items():
                    speed = self.calculate_speed(dataset, data['current_rows'])
                    self.speed_calculators[dataset] = speed
                
                # Display individual download statuses
                for dataset, data in progress_data.items():
                    speed = self.speed_calculators.get(dataset, 0)
                    self.display_download_status(dataset, data, speed)
                
                # Display overall summary
                self.display_overall_summary(progress_data)
                
                # Display multi-year projection
                self.display_multi_year_projection()
                
                # Footer
                print(f"\n{Colors.CYAN}{'=' * 80}{Colors.RESET}")
                print(f"{Colors.WHITE}Press Ctrl+C to stop monitoring{Colors.RESET}")
                print(f"{Colors.WHITE}Next refresh in 15 seconds...{Colors.RESET}")
                print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
                
                # Wait for next refresh
                time.sleep(15)
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")
            print(f"{Colors.WHITE}Total monitoring time:{Colors.RESET} {self.format_duration((datetime.now() - self.start_time).total_seconds())}")
        except Exception as e:
            print(f"\n{Colors.RED}Error occurred: {e}{Colors.RESET}")

def main():
    """Main entry point"""
    monitor = UltimatePJMMonitor()
    monitor.run()

if __name__ == "__main__":
    main()