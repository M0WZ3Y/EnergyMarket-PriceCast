# Data Management for Electricity Price Forecasting

## Data Sources

### Primary Sources
1. **PJM Data Miner 2** (dataminer2.pjm.com)
   - Day-Ahead Market hourly LMPs
   - Load data
   - Period: 2016-2024

2. **NOAA** (ncdc.noaa.gov)
   - Weather data (temperature, humidity)
   - Meteorological variables

3. **EIA** (eia.gov/opendata)
   - Fuel prices
   - Renewable energy generation

## Directory Structure

### `raw/`
- Original, unprocessed data from sources
- Maintains data integrity and provenance
- Subdirectories by data source

### `processed/`
- Cleaned and preprocessed data
- Feature engineering results
- Train/validation/test splits

### `external/`
- Backup datasets (ERCOT, Nord Pool)
- Reference data for comparison
- External validation datasets

## Data Processing Pipeline
1. **Collection**: Automated data fetching from APIs
2. **Cleaning**: Missing value handling, outlier detection
3. **Engineering**: Feature creation, lag variables
4. **Validation**: Data quality checks
5. **Splitting**: Time-series aware train/val/test splits

## Expected Data Volume
- ~100,000 rows (hourly data 2016-2024)
- Multiple features per timestamp
- Exogenous variables integration

## Data Quality Considerations
- Missing data protocols
- Outlier detection thresholds
- Consistency checks across sources
- Time zone and timestamp alignment