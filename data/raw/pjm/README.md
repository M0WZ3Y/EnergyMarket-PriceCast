# PJM Data Collection

This directory contains raw data collected from the PJM Data Miner API for electricity market forecasting research.

## Data Sources

The data is collected using the PJM Data Miner tool (https://github.com/rzwink/pjm_dataminer) which provides access to PJM's public APIs.

## Available Data Types

### Primary Target Variables (High Priority)
1. **Settlements Verified Hourly LMPs** (`rt_da_monthly_lmps`) - Most accurate historical price data
2. **Day-Ahead Hourly LMPs** (`da_hrl_lmps`) - Key feature for price forecasting models
3. **Real-Time Hourly LMPs** (`rt_hrl_lmps`) - Primary target variable for hourly forecasting

### Key Features (High Priority)
4. **Hourly Load: Metered** (`hrl_load_metered`) - Actual load consumption data

### Secondary Features (Medium Priority)
5. **Generation by Fuel Type** (`gen_by_fuel`) - Fuel mix of generation resources
6. **Solar Generation** (`solar_gen`) - Hourly solar generation amounts
7. **Wind Generation** (`wind_gen`) - Hourly wind generation amounts

## Data Collection Scripts

### Automated Collection
Use the main collection script to gather all required data:

```bash
cd scripts
python collect_pjm_2014_data.py
```

### Manual Collection
Use the PJM Data Collector directly:

```python
from 03_code.data_pipeline.data_collection.pjm_data_collector import PJMDataCollector

collector = PJMDataCollector()
file_path = collector.collect_data('da_hrl_lmps', output_format='csv')
```

### Direct PJM Tool Usage
For advanced usage, you can use the PJM dataminer tool directly:

```bash
cd pjm_dataminer-master
set PYTHONIOENCODING=utf-8
venv\Scripts\activate
python fetch_pjm.py -u da_hrl_lmps -f csv -o output_file.csv
```

## Data Structure

### LMP Data Columns
- `datetime_beginning_utc`: Timestamp in UTC
- `datetime_beginning_ept`: Timestamp in Eastern Time
- `pnode`: Pricing node identifier
- `pnode_name`: Pricing node name
- `zone`: Zone identifier
- `load_area`: Load area
- `megawatt`: Locational marginal price in $/MWh
- `lamptype`: LMP type (LMP, MCC, MLC, etc.)

### Load Data Columns
- `datetime_beginning_utc`: Timestamp in UTC
- `datetime_beginning_ept`: Timestamp in Eastern Time
- `nerc_region`: NERC region
- `mkt_region`: Market region
- `zone`: Zone identifier
- `load_area`: Load area
- `mw`: Load in megawatts
- `is_verified`: Whether data has been verified

## Historical Data Access

**Important Note**: The PJM API provides access to all available historical data but does not support direct year-based filtering. When collecting data for specific research periods (e.g., 2014), you will need to:

1. Collect all available historical data
2. Filter for the desired time period during preprocessing
3. Store filtered data in the appropriate processed data directory

## Data Processing Pipeline

After collection, data should be processed through the following pipeline:

1. **Raw Data** (`02_data/raw/pjm/`) - Collected from PJM API
2. **Cleaned Data** (`02_data/processed/cleaned/`) - Filtered for 2014, quality checks
3. **Engineered Features** (`02_data/processed/engineered/`) - Feature engineering for ML models
4. **Splits** (`02_data/processed/splits/`) - Train/validation/test splits

## File Naming Convention

- Raw data: `{data_type}_historical_{start_year}_{end_year}.csv`
- Cleaned data: `{data_type}_cleaned_2014.csv`
- Processed data: `{data_type}_processed_2014.csv`

## Data Quality Notes

- **Settlements Verified LMPs**: Most reliable, but may have delays in availability
- **Real-Time LMPs**: May contain preliminary values that get updated
- **Load Data**: Some values may be estimated and later verified
- **Generation Data**: External units may have unknown regional locations

## API Limitations

- Rate limiting may apply for large data requests
- Some historical data may have archiving restrictions
- Real-time data feeds may have limited retention periods
- API requires subscription key (automatically handled by the tool)

## Troubleshooting

### Common Issues

1. **Unicode Encoding Errors**: Ensure `PYTHONIOENCODING=utf-8` is set
2. **Empty Results**: Some data types may not have recent data available
3. **Timeout Errors**: Large datasets may require longer download times
4. **Memory Issues**: Very large datasets may need chunked processing

### Solutions

```bash
# Set encoding before running
export PYTHONIOENCODING=utf-8

# Use virtual environment
cd pjm_dataminer-master
venv\Scripts\activate

# Check available data types first
python fetch_pjm.py --list
```

## References

- PJM Data Miner: https://github.com/rzwink/pjm_dataminer
- PJM Data Miner 2: https://pjm.com/markets-and-operations/etools/data-miner-2.aspx
- PJM API Documentation: Available through the Data Miner portal

## Contact

For issues with data collection, refer to the PJM Data Miner GitHub repository or PJM support channels.