# Multi-Year PJM LMP Data Collection Strategy

## Overview

This document outlines the comprehensive strategy for collecting PJM Locational Marginal Price (LMP) data for multiple years (2014-2024) to support electricity market forecasting research.

## Datasets to Collect

### 1. Day-Ahead Hourly LMPs (`da_hrl_lmps`)
- **Description**: Hourly Day-Ahead Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates
- **Importance**: Key feature for price forecasting models
- **Size**: ~339,648 rows per year (~32.4 MB)
- **Use Case**: Forward-looking energy price analysis, covers all bus locations and aggregates in PJM region

### 2. Settlements Verified Hourly LMPs (`rt_da_monthly_lmps`)
- **Description**: Verified hourly Real-Time LMPs for aggregate and zonal pnodes used in settlements and final Day-Ahead LMPs
- **Importance**: Most accurate historical price data
- **Size**: ~242,544 rows per year (~23.1 MB)
- **Use Case**: Settlement-quality data for financial analysis, includes both Real-Time and Day-Ahead LMPs

## Collection Strategy

### Phase 1: Current Downloads (2014)
**Status**: Currently in progress
- Day-Ahead Hourly LMPs: ~27,000/339,648 rows downloaded (8%)
- Settlements Verified Hourly LMPs: ~12,000/242,544 rows downloaded (5%)

**Processing Script**: `scripts/process_current_downloads.py`
- Monitors download completion
- Filters data for 2014
- Saves to organized directory structure
- Generates detailed summaries

### Phase 2: Multi-Year Collection (2014-2024)
**Script**: `scripts/collect_multi_year_lmp_data.py`

**Features**:
- Automated collection for all years (2014-2024)
- Separate files for each year to maintain data integrity
- Comprehensive error handling and logging
- Progress tracking and summary generation
- Automatic directory structure creation

**Directory Structure**:
```
02_data/raw/pjm/
├── day_ahead_hourly_lmps/
│   ├── 2014/
│   │   └── da_hrl_lmps_2014.csv
│   ├── 2015/
│   │   └── da_hrl_lmps_2015.csv
│   └── ...
│   └── 2024/
│       └── da_hrl_lmps_2024.csv
├── settlement_verified_hourly_lmps/
│   ├── 2014/
│   │   └── rt_da_monthly_lmps_2014.csv
│   ├── 2015/
│   │   └── rt_da_monthly_lmps_2015.csv
│   └── ...
│   └── 2024/
│       └── rt_da_monthly_lmps_2024.csv
```

## Data Processing Pipeline

### 1. Download Phase
- Uses PJM dataminer API with proper authentication
- Handles large datasets with chunked processing
- Implements retry logic for network issues
- Monitors progress and provides status updates

### 2. Processing Phase
- Filters data by year using `datetime_beginning_ept` column
- Analyzes market run types (Real-Time vs Day-Ahead)
- Generates comprehensive summaries for each dataset
- Validates data integrity and completeness

### 3. Quality Assurance
- Checks for missing data periods
- Validates date ranges and continuity
- Analyzes market run type distributions
- Generates detailed reports for each year

## Usage Instructions

### Step 1: Wait for Current Downloads to Complete
Monitor the current downloads using any of these tools:
- `scripts/monitor_downloads.py` - Basic monitoring
- `scripts/enhanced_pjm_download_monitor.py` - Enhanced monitoring with zero-traffic detection
- `scripts/pjm_download_status.py` - Comprehensive status monitoring

### Step 2: Process Current 2014 Downloads
Once downloads complete, run:
```bash
python scripts/process_current_downloads.py
```

This will:
- Verify download completion
- Filter data for 2014
- Save processed files to appropriate directories
- Generate detailed summaries

### Step 3: Start Multi-Year Collection
After processing 2014 data, run:
```bash
python scripts/collect_multi_year_lmp_data.py
```

This will:
- Collect data for all years 2014-2024
- Create separate files for each year
- Generate comprehensive master summary
- Provide progress tracking throughout the process

## Expected Timeline

### Current Downloads (2014)
- Estimated completion: 2-3 hours
- Day-Ahead LMPs: ~1.9 hours remaining
- Settlement LMPs: ~3.1 hours remaining

### Multi-Year Collection (2015-2024)
- Estimated time: 20-30 hours total
- 2-3 hours per year per dataset
- Can be run unattended with comprehensive logging

## Data Quality and Validation

### Automated Checks
- Date range validation for each year
- Market run type distribution analysis
- Missing data period detection
- File size and row count verification

### Manual Validation
- Sample data inspection
- Cross-year consistency checks
- Market event verification
- Price range reasonableness checks

## Research Applications

### Price Forecasting Models
- **Training Data**: 2014-2021 (8 years)
- **Validation Data**: 2022-2023 (2 years)
- **Test Data**: 2024 (1 year)

### Analysis Types
- Hourly price forecasting
- Volatility analysis
- Market run type comparison
- Seasonal pattern analysis
- Geographic location analysis

## Storage Requirements

### Raw Data
- Day-Ahead LMPs: ~357 MB total (11 years × ~32.4 MB)
- Settlement LMPs: ~254 MB total (11 years × ~23.1 MB)
- **Total**: ~611 MB for complete dataset

### Processed Data
- Additional ~100 MB for summaries and reports
- **Grand Total**: ~711 MB for all years

## Troubleshooting

### Common Issues
1. **Download Interruptions**: Scripts include retry logic and resume capability
2. **Memory Issues**: Chunked processing handles large files efficiently
3. **API Rate Limits**: Built-in delays between requests prevent rate limiting
4. **File Path Issues**: All scripts handle Windows path formatting properly

### Support Scripts
- `scripts/monitor_downloads.py` - Real-time download monitoring
- `scripts/process_current_downloads.py` - Process completed downloads
- `scripts/collect_multi_year_lmp_data.py` - Full multi-year collection

## Next Steps

1. **Immediate**: Monitor current 2014 downloads to completion
2. **Short-term**: Process 2014 data and validate quality
3. **Medium-term**: Execute multi-year collection for 2015-2024
4. **Long-term**: Implement feature engineering and model development pipeline

## Contact and Support

For issues or questions about the data collection process:
- Check log files: `multi_year_pjm_collection.log`, `process_current_downloads.log`
- Review generated summaries in each dataset directory
- Consult the master summary: `MULTI_YEAR_LMP_DATA_COLLECTION_SUMMARY.md`

---

**Last Updated**: 2025-11-26
**Status**: Phase 1 in progress (2014 downloads)
**Next Milestone**: Complete 2014 downloads and processing