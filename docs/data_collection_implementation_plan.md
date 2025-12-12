# Data Collection Pipeline Implementation Plan

## 📋 Overview

This document provides the detailed implementation plan for the data collection pipeline, including configuration files, code specifications, and step-by-step implementation instructions.

## 🎯 Implementation Priority

### Phase 1: Core Infrastructure (Week 1-2)
1. Configuration management system
2. PJM data collector (highest priority)
3. Data validation framework
4. Basic storage system

### Phase 2: Extended Data Sources (Week 3-4)
1. NOAA weather data collector
2. EIA data collector
3. Advanced data processing
4. Feature engineering pipeline

### Phase 3: Optimization & Monitoring (Week 5-6)
1. Performance optimization
2. Quality monitoring system
3. Error handling and recovery
4. Documentation and testing

## 📁 Configuration Files

### 1. Data Collection Configuration

**File:** `06_deployment/config/data_collection_config.yaml`

```yaml
# Data Collection Pipeline Configuration
data_sources:
  pjm:
    base_url: "https://api.pjm.com/api/v1/"
    authentication:
      api_key: "${PJM_API_KEY}"
      timeout: 300
      retry_attempts: 3
      retry_delay: 5
    
    data_types:
      real_time_hourly_lmp:
        endpoint: "rt_hrl_lmps"
        params:
          filetype: "json"
          startdate: "2014-01-01"
          enddate: "2024-12-31"
          fields: "datetime,version,version_status,pnode_id,pnode_name,total_lmp_rt,energy_lmp_rt,congestion_lmp_rt,marginal_loss_lmp_rt"
        
      day_ahead_hourly_lmp:
        endpoint: "da_hrl_lmps"
        params:
          filetype: "json"
          startdate: "2014-01-01"
          enddate: "2024-12-31"
          fields: "datetime,version,pnode_id,pnode_name,total_lmp_da,energy_lmp_da,congestion_lmp_da,marginal_loss_lmp_da"
        
      settlements_hourly_lmp:
        endpoint: "settled_data"
        params:
          filetype: "json"
          startdate: "2014-01-01"
          enddate: "2024-12-31"
          fields: "datetime,pnode_id,pnode_name,total_lmp,energy_lmp,congestion_lmp,marginal_loss_lmp"
        
      instantaneous_load:
        endpoint: "load"
        params:
          filetype: "json"
          startdate: "2014-01-01"
          enddate: "2024-12-31"
          fields: "datetime,area,area_type,load_mw"
        
      load_forecast:
        endpoint: "load_forecast"
        params:
          filetype: "json"
          forecast_date: "current"
          fields: "forecast_date,forecast_hour,area,area_type,load_mw"
        
      generation_fuel_type:
        endpoint: "gen_by_fuel"
        params:
          filetype: "json"
          startdate: "2014-01-01"
          enddate: "2024-12-31"
          fields: "datetime,fuel_type,gen_mw"
    
    collection_settings:
      batch_size: 365  # days per request
      parallel_requests: 4
      update_frequency: "hourly"
      data_retention_days: 3650

  noaa:
    base_url: "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
    authentication:
      api_key: "${NOAA_API_KEY}"
      timeout: 300
      retry_attempts: 3
      retry_delay: 5
    
    stations:
      - station_id: "KNYC"
        name: "New York Central Park"
        latitude: 40.779
        longitude: -73.969
      - station_id: "KPHL"
        name: "Philadelphia International"
        latitude: 39.874
        longitude: -75.241
      - station_id: "KDCA"
        name: "Washington Reagan"
        latitude: 38.851
        longitude: -77.040
      - station_id: "KBWI"
        name: "Baltimore Washington"
        latitude: 39.175
        longitude: -76.668
      - station_id: "KCLE"
        name: "Cleveland Hopkins"
        latitude: 41.411
        longitude: -81.849
    
    data_types:
      hourly:
        datasetid: "GHCND"
        datatypeid:
          - "TMP"  # Temperature
          - "DEW"  # Dew point
          - "SLP"  # Sea level pressure
          - "WDSP"  # Wind speed
          - "WDIR"  # Wind direction
          - "PRCP"  # Precipitation
          - "SNWD"  # Snow depth
        units: "metric"
        limit: 1000
      
      daily:
        datasetid: "GHCND"
        datatypeid:
          - "TMAX"  # Max temperature
          - "TMIN"  # Min temperature
          - "PRCP"  # Precipitation
          - "SNOW"  # Snowfall
        units: "metric"
        limit: 1000
    
    collection_settings:
      batch_size: 365  # days per request
      parallel_requests: 2
      update_frequency: "daily"
      data_retention_days: 3650

  eia:
    base_url: "https://api.eia.gov/v2/"
    authentication:
      api_key: "${EIA_API_KEY}"
      timeout: 300
      retry_attempts: 3
      retry_delay: 5
    
    data_types:
      fuel_prices:
        endpoint: "electricity/rto/region-type-data/data/"
        params:
          frequency: "daily"
          data: "price"
          facets:
            region: [
              "CAL", "CAR", "CENT", "FLA", "MIDA", "MIDW", 
              "NE", "NY", "SE", "SW", "TEN", "TEX"
            ]
          start: "2014-01-01"
          end: "2024-12-31"
          sort:
            column: "period"
            direction: "desc"
          offset: 0
          length: 5000
      
      renewable_generation:
        endpoint: "electricity/electric-power-operational-data/data/"
        params:
          frequency: "monthly"
          data: "generation"
          facets:
            fueltypeid: ["WND", "SUN", "HYC", "GEO", "OTH"]
            sectorid: ["99"]
          start: "2014-01-01"
          end: "2024-12-31"
          sort:
            column: "period"
            direction: "desc"
          offset: 0
          length: 5000
    
    collection_settings:
      batch_size: 365  # days per request
      parallel_requests: 2
      update_frequency: "daily"
      data_retention_days: 3650

storage:
  paths:
    raw_data: "02_data/raw/"
    processed_data: "02_data/processed/"
    features: "02_data/features/"
    metadata: "02_data/metadata/"
    logs: "02_data/logs/"
  
  formats:
    raw: "parquet"
    processed: "parquet"
    features: "parquet"
    metadata: "json"
  
  compression:
    enabled: true
    algorithm: "gzip"
    level: 6
  
  partitioning:
    enabled: true
    scheme: "by_date_and_source"
    format: "{source}/{data_type}/{year}/{month}/{day}/"

quality_control:
  validation_rules:
    completeness_threshold: 0.95
    outlier_detection:
      method: "iqr"
      threshold: 1.5
    temporal_consistency:
      max_gap_hours: 24
      max_spike_factor: 10.0
  
  alerts:
    enabled: true
    email_recipients: ["researcher@university.edu"]
    slack_webhook: "${SLACK_WEBHOOK_URL}"
    thresholds:
      completeness_drop: 0.1
      error_rate: 0.05

processing:
  feature_engineering:
    temporal_features:
      - "hour_of_day"
      - "day_of_week"
      - "month_of_year"
      - "season"
      - "holiday_indicator"
      - "weekend_indicator"
    
    lag_features:
      - "price_lag_1h"
      - "price_lag_24h"
      - "price_lag_168h"
      - "load_lag_1h"
      - "load_lag_24h"
    
    rolling_features:
      - "price_mean_24h"
      - "price_std_24h"
      - "load_mean_24h"
      - "load_std_24h"
    
    weather_features:
      - "temperature_squared"
      - "humidity_temperature_interaction"
      - "wind_speed_squared"
  
  data_splitting:
    training_ratio: 0.7
    validation_ratio: 0.15
    test_ratio: 0.15
    method: "temporal"
    
scheduling:
  jobs:
    pjm_real_time:
      schedule: "0 * * * *"  # Every hour
      collector: "pjm_collector"
      data_types: ["real_time_hourly_lmp", "instantaneous_load"]
      
    pjm_daily:
      schedule: "0 2 * * *"  # Daily at 2 AM
      collector: "pjm_collector"
      data_types: ["day_ahead_hourly_lmp", "settlements_hourly_lmp", "load_forecast"]
      
    noaa_daily:
      schedule: "0 3 * * *"  # Daily at 3 AM
      collector: "noaa_collector"
      data_types: ["hourly", "daily"]
      
    eia_daily:
      schedule: "0 4 * * *"  # Daily at 4 AM
      collector: "eia_collector"
      data_types: ["fuel_prices", "renewable_generation"]
      
    processing:
      schedule: "0 5 * * *"  # Daily at 5 AM
      processor: "data_processor"
      operations: ["validation", "cleaning", "feature_engineering"]

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    file:
      enabled: true
      filename: "02_data/logs/data_collection.log"
      max_size: "100MB"
      backup_count: 5
    console:
      enabled: true
    syslog:
      enabled: false
```

### 2. Data Validation Rules Configuration

**File:** `03_code/data_collection/config/validation_rules.yaml`

```yaml
# Data Validation Rules Configuration
validation_rules:
  pjm:
    real_time_hourly_lmp:
      required_columns:
        - "datetime"
        - "pnode_id"
        - "total_lmp_rt"
      
      column_types:
        datetime: "datetime64[ns]"
        pnode_id: "string"
        total_lmp_rt: "float64"
      
      value_ranges:
        total_lmp_rt:
          min: -100
          max: 1000
        energy_lmp_rt:
          min: -100
          max: 1000
        congestion_lmp_rt:
          min: -500
          max: 500
        marginal_loss_lmp_rt:
          min: -100
          max: 100
      
      temporal_checks:
        max_gap_minutes: 60
        expected_frequency: "1H"
        duplicate_tolerance: 0
    
    instantaneous_load:
      required_columns:
        - "datetime"
        - "area"
        - "load_mw"
      
      column_types:
        datetime: "datetime64[ns]"
        area: "string"
        load_mw: "float64"
      
      value_ranges:
        load_mw:
          min: 0
          max: 200000
      
      temporal_checks:
        max_gap_minutes: 5
        expected_frequency: "5min"
        duplicate_tolerance: 0

  noaa:
    hourly:
      required_columns:
        - "datetime"
        - "station"
        - "datatype"
        - "value"
      
      column_types:
        datetime: "datetime64[ns]"
        station: "string"
        datatype: "string"
        value: "float64"
      
      value_ranges:
        TMP:  # Temperature
          min: -50
          max: 60
        DEW:  # Dew point
          min: -50
          max: 50
        PRCP:  # Precipitation
          min: 0
          max: 500
        WDSP:  # Wind speed
          min: 0
          max: 100
      
      temporal_checks:
        max_gap_minutes: 60
        expected_frequency: "1H"
        duplicate_tolerance: 0

  eia:
    fuel_prices:
      required_columns:
        - "period"
        - "region"
        - "price"
      
      column_types:
        period: "datetime64[ns]"
        region: "string"
        price: "float64"
      
      value_ranges:
        price:
          min: 0
          max: 1000
      
      temporal_checks:
        max_gap_hours: 24
        expected_frequency: "1D"
        duplicate_tolerance: 0

quality_metrics:
  completeness:
    threshold: 0.95
    weight: 0.3
  
  accuracy:
    threshold: 0.90
    weight: 0.3
  
  consistency:
    threshold: 0.90
    weight: 0.2
  
  timeliness:
    threshold: 0.95
    weight: 0.2
```

## 🔧 Code Implementation Specifications

### 1. PJM Data Collector

**File:** `03_code/data_collection/collectors/pjm_collector.py`

```python
"""
PJM Data Collector
Collects real-time and historical data from PJM API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from ..utils.config_manager import ConfigManager
from ..utils.logger import setup_logger

class PJMDataCollector:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = setup_logger(__name__)
        self.base_url = config.get('data_sources.pjm.base_url')
        self.api_key = config.get('data_sources.pjm.authentication.api_key')
        self.session = requests.Session()
        self.session.headers.update({
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/json'
        })
    
    def collect_real_time_hourly_lmp(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect real-time hourly LMP data"""
        endpoint = "rt_hrl_lmps"
        params = {
            'filetype': 'json',
            'startdate': start_date,
            'enddate': end_date,
            'fields': 'datetime,version,version_status,pnode_id,pnode_name,total_lmp_rt,energy_lmp_rt,congestion_lmp_rt,marginal_loss_lmp_rt'
        }
        
        return self._fetch_data(endpoint, params)
    
    def collect_day_ahead_hourly_lmp(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect day-ahead hourly LMP data"""
        endpoint = "da_hrl_lmps"
        params = {
            'filetype': 'json',
            'startdate': start_date,
            'enddate': end_date,
            'fields': 'datetime,version,pnode_id,pnode_name,total_lmp_da,energy_lmp_da,congestion_lmp_da,marginal_loss_lmp_da'
        }
        
        return self._fetch_data(endpoint, params)
    
    def collect_settlements_hourly_lmp(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect settlements verified hourly LMP data"""
        endpoint = "settled_data"
        params = {
            'filetype': 'json',
            'startdate': start_date,
            'enddate': end_date,
            'fields': 'datetime,pnode_id,pnode_name,total_lmp,energy_lmp,congestion_lmp,marginal_loss_lmp'
        }
        
        return self._fetch_data(endpoint, params)
    
    def collect_instantaneous_load(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect instantaneous load data"""
        endpoint = "load"
        params = {
            'filetype': 'json',
            'startdate': start_date,
            'enddate': end_date,
            'fields': 'datetime,area,area_type,load_mw'
        }
        
        return self._fetch_data(endpoint, params)
    
    def collect_load_forecast(self, forecast_date: str = None) -> pd.DataFrame:
        """Collect load forecast data"""
        endpoint = "load_forecast"
        params = {
            'filetype': 'json',
            'forecast_date': forecast_date or 'current',
            'fields': 'forecast_date,forecast_hour,area,area_type,load_mw'
        }
        
        return self._fetch_data(endpoint, params)
    
    def collect_generation_fuel_type(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect generation by fuel type data"""
        endpoint = "gen_by_fuel"
        params = {
            'filetype': 'json',
            'startdate': start_date,
            'enddate': end_date,
            'fields': 'datetime,fuel_type,gen_mw'
        }
        
        return self._fetch_data(endpoint, params)
    
    def _fetch_data(self, endpoint: str, params: Dict) -> pd.DataFrame:
        """Generic method to fetch data from PJM API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                self.logger.warning(f"No data returned from {endpoint}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Data type conversions
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert numeric columns
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            self.logger.info(f"Successfully collected {len(df)} records from {endpoint}")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from {endpoint}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing data from {endpoint}: {str(e)}")
            raise
    
    def collect_bulk_data(self, start_date: str, end_date: str, data_types: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect multiple data types in bulk"""
        results = {}
        
        for data_type in data_types:
            try:
                self.logger.info(f"Collecting {data_type} from {start_date} to {end_date}")
                
                if data_type == 'real_time_hourly_lmp':
                    results[data_type] = self.collect_real_time_hourly_lmp(start_date, end_date)
                elif data_type == 'day_ahead_hourly_lmp':
                    results[data_type] = self.collect_day_ahead_hourly_lmp(start_date, end_date)
                elif data_type == 'settlements_hourly_lmp':
                    results[data_type] = self.collect_settlements_hourly_lmp(start_date, end_date)
                elif data_type == 'instantaneous_load':
                    results[data_type] = self.collect_instantaneous_load(start_date, end_date)
                elif data_type == 'generation_fuel_type':
                    results[data_type] = self.collect_generation_fuel_type(start_date, end_date)
                
            except Exception as e:
                self.logger.error(f"Failed to collect {data_type}: {str(e)}")
                results[data_type] = pd.DataFrame()
        
        return results
```

### 2. Data Validator

**File:** `03_code/data_collection/processors/data_validator.py`

```python
"""
Data Validator
Validates data quality and consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
from ..utils.config_manager import ConfigManager
from ..utils.logger import setup_logger

class DataValidator:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = setup_logger(__name__)
        self.validation_rules = config.load_config('validation_rules.yaml')
    
    def validate_dataframe(self, df: pd.DataFrame, data_type: str, source: str) -> Dict[str, Any]:
        """Validate a DataFrame against predefined rules"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {},
            'summary': {}
        }
        
        try:
            # Get validation rules for this data type
            rules = self.validation_rules['validation_rules'][source].get(data_type, {})
            
            if not rules:
                validation_results['warnings'].append(f"No validation rules found for {source}.{data_type}")
                return validation_results
            
            # Check required columns
            validation_results = self._check_required_columns(df, rules, validation_results)
            
            # Check column types
            validation_results = self._check_column_types(df, rules, validation_results)
            
            # Check value ranges
            validation_results = self._check_value_ranges(df, rules, validation_results)
            
            # Check temporal consistency
            validation_results = self._check_temporal_consistency(df, rules, validation_results)
            
            # Calculate quality metrics
            validation_results['quality_metrics'] = self._calculate_quality_metrics(df)
            
            # Generate summary
            validation_results['summary'] = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'validation_timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'source': source
            }
            
            # Determine overall validity
            validation_results['is_valid'] = len(validation_results['errors']) == 0
            
        except Exception as e:
            self.logger.error(f"Error validating {source}.{data_type}: {str(e)}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _check_required_columns(self, df: pd.DataFrame, rules: Dict, results: Dict) -> Dict:
        """Check if all required columns are present"""
        required_columns = rules.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results
    
    def _check_column_types(self, df: pd.DataFrame, rules: Dict, results: Dict) -> Dict:
        """Check column data types"""
        expected_types = rules.get('column_types', {})
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                try:
                    if expected_type == 'datetime64[ns]':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif expected_type == 'float64':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif expected_type == 'string':
                        df[column] = df[column].astype(str)
                    
                    # Check for conversion errors
                    if df[column].isnull().sum() > 0:
                        null_count = df[column].isnull().sum()
                        warning_msg = f"Column {column} has {null_count} null values after type conversion"
                        results['warnings'].append(warning_msg)
                        self.logger.warning(warning_msg)
                
                except Exception as e:
                    error_msg = f"Error converting column {column} to {expected_type}: {str(e)}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
        
        return results
    
    def _check_value_ranges(self, df: pd.DataFrame, rules: Dict, results: Dict) -> Dict:
        """Check if values are within expected ranges"""
        value_ranges = rules.get('value_ranges', {})
        
        for column, range_config in value_ranges.items():
            if column in df.columns:
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                if min_val is not None:
                    below_min = (df[column] < min_val).sum()
                    if below_min > 0:
                        warning_msg = f"Column {column} has {below_min} values below minimum ({min_val})"
                        results['warnings'].append(warning_msg)
                        self.logger.warning(warning_msg)
                
                if max_val is not None:
                    above_max = (df[column] > max_val).sum()
                    if above_max > 0:
                        warning_msg = f"Column {column} has {above_max} values above maximum ({max_val})"
                        results['warnings'].append(warning_msg)
                        self.logger.warning(warning_msg)
        
        return results
    
    def _check_temporal_consistency(self, df: pd.DataFrame, rules: Dict, results: Dict) -> Dict:
        """Check temporal consistency of time series data"""
        temporal_rules = rules.get('temporal_checks', {})
        
        if 'datetime' not in df.columns:
            return results
        
        # Sort by datetime
        df_sorted = df.sort_values('datetime')
        
        # Check for gaps
        max_gap_minutes = temporal_rules.get('max_gap_minutes')
        if max_gap_minutes:
            time_diffs = df_sorted['datetime'].diff().dropna()
            large_gaps = time_diffs > pd.Timedelta(minutes=max_gap_minutes)
            
            if large_gaps.sum() > 0:
                warning_msg = f"Found {large_gaps.sum()} gaps larger than {max_gap_minutes} minutes"
                results['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
        
        # Check for duplicates
        duplicate_tolerance = temporal_rules.get('duplicate_tolerance', 0)
        duplicates = df_sorted.duplicated(subset=['datetime']).sum()
        
        if duplicates > duplicate_tolerance:
            warning_msg = f"Found {duplicates} duplicate timestamps"
            results['warnings'].append(warning_msg)
            self.logger.warning(warning_msg)
        
        return results
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        metrics = {}
        
        # Completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        metrics['completeness'] = completeness
        
        # Uniqueness (for datetime column if exists)
        if 'datetime' in df.columns:
            unique_timestamps = df['datetime'].nunique()
            total_timestamps = len(df)
            uniqueness = unique_timestamps / total_timestamps if total_timestamps > 0 else 0
            metrics['uniqueness'] = uniqueness
        
        # Validity (percentage of non-null values in numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            valid_numeric = df[numeric_columns].notnull().sum().sum()
            total_numeric = len(df) * len(numeric_columns)
            validity = valid_numeric / total_numeric if total_numeric > 0 else 0
            metrics['validity'] = validity
        
        return metrics
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report"""
        report = []
        report.append("Data Quality Report")
        report.append("=" * 50)
        
        # Summary
        summary = validation_results['summary']
        report.append(f"Data Type: {summary['data_type']}")
        report.append(f"Source: {summary['source']}")
        report.append(f"Total Records: {summary['total_records']}")
        report.append(f"Total Columns: {summary['total_columns']}")
        report.append(f"Validation Timestamp: {summary['validation_timestamp']}")
        report.append(f"Overall Valid: {validation_results['is_valid']}")
        report.append("")
        
        # Quality Metrics
        if validation_results['quality_metrics']:
            report.append("Quality Metrics:")
            for metric, value in validation_results['quality_metrics'].items():
                report.append(f"  {metric}: {value:.3f}")
            report.append("")
        
        # Errors
        if validation_results['errors']:
            report.append("Errors:")
            for error in validation_results['errors']:
                report.append(f"  - {error}")
            report.append("")
        
        # Warnings
        if validation_results['warnings']:
            report.append("Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
            report.append("")
        
        return "\n".join(report)
```

## 🚀 Implementation Steps

### Step 1: Setup Configuration Management
1. Create the configuration files as specified above
2. Set up environment variables for API keys
3. Test configuration loading

### Step 2: Implement PJM Data Collector
1. Create the base collector class
2. Implement specific data type methods
3. Add error handling and retry logic
4. Test with sample data

### Step 3: Implement Data Validation
1. Create the validator class
2. Implement validation rules
3. Add quality metrics calculation
4. Test validation with sample data

### Step 4: Create Storage System
1. Implement raw data storage
2. Create processed data storage
3. Set up feature store
4. Test data retrieval

### Step 5: Integration Testing
1. Test end-to-end pipeline
2. Validate data quality
3. Performance testing
4. Error handling testing

## 📊 Success Criteria

### Functional Requirements
- [ ] Successfully collect all 7 PJM data types
- [ ] Validate data quality with >95% completeness
- [ ] Store data in efficient format (Parquet)
- [ ] Generate quality reports
- [ ] Handle errors gracefully

### Performance Requirements
- [ ] Collect 1 year of data in < 30 minutes
- [ ] Process data with < 5% CPU usage
- [ ] Store data with < 50% compression ratio
- [ ] Validate data with < 1 minute per dataset

### Quality Requirements
- [ ] Zero data loss during collection
- [ ] Comprehensive error logging
- [ ] Automated quality monitoring
- [ ] Reproducible results

This implementation plan provides a solid foundation for building a robust data collection pipeline for your electricity price forecasting thesis.