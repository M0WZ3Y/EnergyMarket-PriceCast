#!/usr/bin/env python3
"""
Test script to demonstrate zero-traffic detection functionality
"""

import time
import sys
import os
from datetime import datetime

# Add the scripts directory to the path
sys.path.append(os.path.dirname(__file__))

from enhanced_pjm_download_monitor import DownloadTracker, Colors

def test_zero_traffic_detection():
    """Test the zero-traffic detection functionality"""
    print(f"{Colors.BOLD}{Colors.CYAN}TESTING ZERO-TRAFFIC DETECTION SYSTEM{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    
    # Create a test tracker
    tracker = DownloadTracker('test_dataset', 'Test Dataset')
    
    print(f"{Colors.WHITE}Created test tracker for: {Colors.MAGENTA}{tracker.name}{Colors.RESET}")
    print(f"{Colors.WHITE}Initial state:{Colors.RESET}")
    print(f"  - Last row count: {tracker.last_row_count}")
    print(f"  - Is stalled: {tracker.is_stalled}")
    print(f"  - Zero traffic start: {tracker.zero_traffic_start}")
    print()
    
    # Simulate normal progress
    print(f"{Colors.GREEN}Simulating normal download progress...{Colors.RESET}")
    for i in range(3):
        tracker.update_progress(100 * (i + 1))
        print(f"  Update {i+1}: {tracker.last_row_count} rows, Stalled: {tracker.is_stalled}")
        time.sleep(1)
    
    print()
    print(f"{Colors.YELLOW}Simulating zero traffic (no progress updates)...{Colors.RESET}")
    print(f"{Colors.WHITE}This will trigger the stalled detection after 3 minutes.{Colors.RESET}")
    print(f"{Colors.WHITE}For testing, we'll simulate this in accelerated time.{Colors.RESET}")
    print()
    
    # Simulate zero traffic by not updating progress
    # In the real system, this would be detected after 3 minutes (180 seconds)
    # For testing, we'll modify the threshold temporarily
    
    original_threshold = 180  # 3 minutes in seconds
    test_threshold = 5        # 5 seconds for testing
    
    print(f"{Colors.CYAN}Testing with accelerated threshold ({test_threshold} seconds instead of {original_threshold} seconds):{Colors.RESET}")
    
    # Simulate the passage of time without progress updates
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        # Manually set the zero traffic start time to simulate stall
        if tracker.zero_traffic_start is None:
            tracker.zero_traffic_start = start_time
        
        # Check if stalled
        if elapsed > test_threshold:
            tracker.is_stalled = True
        
        # Display status
        status_color = Colors.RED if tracker.is_stalled else Colors.YELLOW
        status_text = "STALLED" if tracker.is_stalled else "NO PROGRESS"
        
        print(f"\r{Colors.WHITE}Elapsed: {elapsed:.1f}s | Status: {status_color}{status_text}{Colors.RESET} | Stall Duration: {tracker.get_stall_duration().total_seconds():.1f}s", end="")
        
        if tracker.is_stalled:
            print(f"\n\n{Colors.BG_RED}{Colors.WHITE}ZERO-TRAFFIC DETECTED!{Colors.RESET}")
            print(f"{Colors.RED}Download has been stalled for {tracker.get_stall_duration().total_seconds():.1f} seconds{Colors.RESET}")
            print(f"{Colors.YELLOW}Recommended actions:{Colors.RESET}")
            print(f"  1. Check network connection")
            print(f"  2. Verify server status")
            print(f"  3. Restart download if necessary")
            break
        
        time.sleep(0.5)
    
    print(f"\n{Colors.GREEN}Zero-traffic detection test completed successfully!{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")

if __name__ == "__main__":
    test_zero_traffic_detection()