# EIA Electricity API Subcategories Selection

## 🎯 **ELECTRICITY API SUBCATEGORIES PRIORITY**

Based on the available Electricity API subcategories, here's the prioritized selection for your electricity price forecasting thesis:

---

## 🔴 **CRITICAL PRIORITY (Must Select)**

### **1. Electric Power Operations (Daily and Hourly)**
**Why Critical**: Provides high-frequency generation and operational data that directly impacts electricity prices

**Key Data to Collect**:
- Hourly electricity generation by fuel type
- Daily generation patterns
- Real-time operational data
- Fuel-specific generation metrics

**Importance**: This is the most valuable for price forecasting as it provides the high-frequency supply-side data that drives price formation.

### **2. Electric Power Operations (Annual and Monthly)**
**Why Critical**: Provides comprehensive generation trends and seasonal patterns

**Key Data to Collect**:
- Monthly electricity generation by fuel type
- Annual generation capacity and output
- Seasonal generation patterns
- Long-term fuel mix trends

**Importance**: Complements daily/hourly data with longer-term trends and seasonal patterns.

---

## 🔶 **HIGH PRIORITY (Should Select)**

### **3. State Specific Data**
**Why Important**: Provides state-level granularity for PJM territory analysis

**Key Data to Collect**:
- State-level electricity generation
- State-level consumption patterns
- State-specific fuel mix
- Regional price differentials

**Importance**: Essential for capturing geographic variations within PJM territory.

### **4. Inventory of Operable Generators**
**Why Important**: Provides capacity constraints and generation asset information

**Key Data to Collect**:
- Generator capacity by fuel type
- Generator locations and status
- Retirement and addition schedules
- Technology-specific characteristics

**Importance**: Helps understand supply constraints and capacity limitations that affect prices.

---

## 🔹 **MEDIUM PRIORITY (Optional)**

### **5. Electricity Sales to Ultimate Customers**
**Why Useful**: Provides demand-side information and retail price context

**Key Data to Collect**:
- Retail electricity sales by sector
- Customer consumption patterns
- Retail price trends
- Demand growth metrics

**Importance**: Useful for demand-side analysis but less critical for wholesale price forecasting.

### **6. Electric Power Operations for Individual Power Plant**
**Why Optional**: Very granular data, may be too detailed for thesis scope

**Key Data to Collect**:
- Individual plant generation data
- Plant-specific operational metrics
- Plant outage schedules
- Plant efficiency data

**Importance**: Most granular data available, but may be overly detailed for price forecasting models.

---

## 📊 **DETAILED SELECTION STRATEGY**

### **Phase 1: Core Operations Data (Week 1) - Foundation**

#### **Day 1-2: Electric Power Operations (Daily and Hourly)**
**Objective**: Establish high-frequency generation baseline

**Data Collection Strategy**:
- **Frequency Priority**: Hourly > Daily
- **Fuel Type Focus**: Natural Gas (NG), Coal (COAL), Solar (SUN), Wind (WND)
- **Geographic Focus**: PJM core states (PA, NJ, MD, DE, VA)
- **Time Period**: 2014-2024 (10 years)
- **API Parameters**:
  ```python
  frequency: "hourly"
  data: ["generation"]
  facets: {
    "fueltypeid": ["NG", "COAL", "SUN", "WND"],
    "stateid": ["PA", "NJ", "MD", "DE", "VA"]
  }
  ```

**Expected Data Volume**: ~876,000 records per fuel type per year
**Storage Strategy**: Partition by year/fuel_type/state
**Quality Checks**: Timestamp continuity, value ranges, missing data

#### **Day 3-4: Electric Power Operations (Annual and Monthly)**
**Objective**: Add seasonal and long-term trends

**Data Collection Strategy**:
- **Frequency Priority**: Monthly > Annual
- **All Fuel Types**: Complete fuel mix analysis
- **Extended Geography**: All PJM states (WV, OH, NC)
- **API Parameters**:
  ```python
  frequency: "monthly"
  data: ["generation", "consumption"]
  facets: {
    "stateid": ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "NC"]
  }
  ```

**Integration Strategy**: Align with hourly data for comprehensive view
**Validation**: Cross-check monthly totals with hourly aggregations

### **Phase 2: Regional Analysis (Week 2) - Geographic Context**

#### **Day 5-6: State Specific Data**
**Objective**: Capture regional variations and state-level dynamics

**Data Collection Strategy**:
- **State-Level Granularity**: Individual state analysis
- **Multiple Metrics**: Generation, consumption, sales, prices
- **Historical Trends**: Multi-year comparisons
- **API Parameters**:
  ```python
  frequency: "monthly"
  data: ["generation", "consumption", "sales", "price"]
  facets: {
    "stateid": ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "NC"]
  }
  ```

**Analysis Opportunities**:
- State-specific renewable penetration rates
- Regional price differentials
- Load center vs generation center patterns
- Cross-border electricity flows

#### **Day 7: Inventory of Operable Generators**
**Objective**: Understand capacity constraints and infrastructure

**Data Collection Strategy**:
- **Annual Snapshots**: Year-by-year capacity changes
- **Technology Breakdown**: Fuel type, technology, age
- **Operational Status**: Active, retired, planned units
- **API Parameters**:
  ```python
  frequency: "annual"
  data: ["capacity", "status", "technology"]
  facets: {
    "stateid": ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "NC"]
  }
  ```

**Strategic Value**:
- Capacity constraint identification
- Retirement schedule impact
- New capacity additions
- Technology transition tracking

### **Phase 3: Demand Context (Week 3) - Market Dynamics**

#### **Day 8-9: Electricity Sales to Ultimate Customers**
**Objective**: Add demand-side perspective

**Data Collection Strategy**:
- **Sector Breakdown**: Residential, Commercial, Industrial
- **Sales Volume**: MWh consumption by sector
- **Price Context**: Retail price trends
- **API Parameters**:
  ```python
  frequency: "monthly"
  data: ["sales", "customers", "revenue"]
  facets: {
    "sectorid": ["RES", "COM", "IND"],
    "stateid": ["PA", "NJ", "MD", "DE", "VA"]
  }
  ```

**Integration Benefits**:
- Demand elasticity analysis
- Sector-specific consumption patterns
- Retail vs wholesale price relationships
- Economic activity indicators

#### **Day 10: Electric Power Operations for Individual Power Plant**
**Objective**: Detailed plant-level analysis (Optional)

**Data Collection Strategy**:
- **Plant-Level Detail**: Individual generator performance
- **Operational Metrics**: Heat rates, efficiency, outage patterns
- **Strategic Plants**: Largest, most critical units
- **API Parameters**:
  ```python
  frequency: "monthly"
  data: ["generation", "heat_rate", "capacity_factor"]
  facets: {
    "plantid": ["selected_major_plants"]
  }
  ```

**Research Value**:
- Plant-specific outage impacts
- Efficiency trend analysis
- Benchmarking studies
- Investment decision support

### **Phase 4: Integration and Validation (Week 4) - Quality Assurance**

#### **Day 11-12: Data Integration**
**Objective**: Create unified dataset

**Integration Tasks**:
- **Temporal Alignment**: Synchronize all datasets to common timestamps
- **Geographic Consistency**: Ensure state/region coding consistency
- **Unit Standardization**: Convert all measurements to consistent units
- **Data Validation**: Cross-check overlapping data sources

#### **Day 13-14: Quality Assurance**
**Objective**: Ensure data reliability

**Quality Checks**:
- **Completeness**: >95% data coverage requirement
- **Accuracy**: Range validation, outlier detection
- **Consistency**: Cross-validation between datasets
- **Timeliness**: Real-time vs historical data alignment

---

## 🎯 **IMPLEMENTATION TIMELINE**

### **Week 1: Foundation Building**
- **Monday-Tuesday**: Daily/Hourly operations data collection
- **Wednesday-Thursday**: Annual/Monthly operations data
- **Friday**: Initial data validation and cleaning

### **Week 2: Geographic Expansion**
- **Monday-Tuesday**: State-specific data collection
- **Wednesday**: Generator inventory data
- **Thursday-Friday**: Regional analysis and integration

### **Week 3: Market Context**
- **Monday-Tuesday**: Sales data collection
- **Wednesday**: Plant-level data (optional)
- **Thursday-Friday**: Demand analysis integration

### **Week 4: Quality and Integration**
- **Monday-Tuesday**: Data integration and standardization
- **Wednesday-Thursday**: Quality assurance and validation
- **Friday**: Final dataset preparation

---

## 📊 **DATA COLLECTION METRICS**

### **Volume Estimates**
- **Hourly Data**: ~350,000 records per fuel type per year
- **Monthly Data**: ~12,000 records per state per year
- **Annual Data**: ~1,000 records per state per year
- **Total Expected**: ~5-10 million records

### **Storage Requirements**
- **Raw Data**: ~10-20 GB (JSON/CSV format)
- **Processed Data**: ~2-5 GB (Parquet format)
- **Feature Store**: ~1-2 GB (Optimized features)

### **Processing Time**
- **API Calls**: 2-4 hours per dataset
- **Data Cleaning**: 4-8 hours per dataset
- **Integration**: 8-12 hours total
- **Validation**: 4-6 hours total

---

## 🔧 **TECHNICAL IMPLEMENTATION**

### **API Rate Management**
- **Request Limits**: 1000 requests/hour
- **Batch Processing**: Collect in 1-year chunks
- **Retry Logic**: Exponential backoff for failures
- **Parallel Processing**: Multiple fuel types simultaneously

### **Error Handling**
- **Missing Data**: Interpolation strategies
- **API Failures**: Retry with exponential backoff
- **Data Validation**: Range checks and outlier detection
- **Logging**: Comprehensive error tracking

### **Performance Optimization**
- **Caching**: Store API responses locally
- **Compression**: Use gzip for data transfer
- **Batching**: Combine multiple requests
- **Async Processing**: Parallel data collection

---

## 🎯 **SPECIFIC DATASETS TO TARGET**

### **Electric Power Operations (Daily and Hourly)**
```
/electricity/electric-power-operational-data/data/
- frequency: hourly, daily
- data: generation
- facets: 
  - fueltypeid: NG (Natural Gas), COAL (Coal), SUN (Solar), WND (Wind)
  - stateid: PA, NJ, MD, DE, VA, WV, OH, NC
- period: 2014-01-01 to 2024-12-31
```

### **Electric Power Operations (Annual and Monthly)**
```
/electricity/electric-power-operational-data/data/
- frequency: monthly, annual
- data: generation, consumption
- facets:
  - fueltypeid: All fuel types
  - stateid: PJM states
- period: 2014-01-01 to 2024-12-31
```

### **State Specific Data**
```
/electricity/state/data/
- frequency: monthly, annual
- data: generation, consumption, sales
- facets:
  - stateid: PA, NJ, MD, DE, VA, WV, OH, NC
- period: 2014-01-01 to 2024-12-31
```

### **Inventory of Operable Generators**
```
/electricity/generator-inventory/data/
- frequency: annual
- data: capacity, status
- facets:
  - stateid: PJM states
  - fueltypeid: All fuel types
- period: 2014-01-01 to 2024-12-31
```

---

## 🔧 **API CALL EXAMPLES**

### **Hourly Generation by Fuel Type**
```python
https://api.eia.gov/v2/electricity/electric-power-operational-data/data/?frequency=hourly&data[0]=generation&facets[fueltypeid][]=NG&facets[stateid][]=PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

### **Monthly Generation by State**
```python
https://api.eia.gov/v2/electricity/electric-power-operational-data/data/?frequency=monthly&data[0]=generation&facets[stateid][]=PA&facets[stateid][]=NJ&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

### **Generator Inventory**
```python
https://api.eia.gov/v2/electricity/generator-inventory/data/?frequency=annual&data[0]=capacity&facets[stateid][]=PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

---

## 📋 **SELECTION CHECKLIST**

### **Must-Select Subcategories**
- [ ] **Electric Power Operations (Daily and Hourly)** ✅
- [ ] **Electric Power Operations (Annual and Monthly)** ✅

### **Should-Select Subcategories**
- [ ] **State Specific Data** ✅
- [ ] **Inventory of Operable Generators** ✅

### **Optional Subcategories**
- [ ] **Electricity Sales to Ultimate Customers** (if time permits)
- [ ] **Electric Power Operations for Individual Power Plant** (if detailed analysis needed)

---

## 🎯 **PRIORITY ORDER FOR SELECTION**

### **1. Start Here (Most Critical)**
**Electric Power Operations (Daily and Hourly)**
- This provides the high-frequency generation data
- Direct impact on electricity prices
- Essential for hourly price forecasting

### **2. Second Priority**
**Electric Power Operations (Annual and Monthly)**
- Provides seasonal patterns
- Complements daily/hourly data
- Important for trend analysis

### **3. Third Priority**
**State Specific Data**
- Geographic granularity for PJM
- Regional variations in generation
- State-specific fuel mix

### **4. Fourth Priority**
**Inventory of Operable Generators**
- Capacity constraints
- Supply-side limitations
- Infrastructure analysis

### **5. Optional (Time Permitting)**
**Electricity Sales to Ultimate Customers**
- Demand-side context
- Retail price trends
- Consumption patterns

### **6. Last Priority**
**Electric Power Operations for Individual Power Plant**
- Most granular data
- May be overly detailed
- Use only for specific plant analysis

---

## 🚀 **IMPLEMENTATION STRATEGY**

### **Week 1: Core Operations Data**
- Focus on Daily and Hourly operations
- Collect generation by fuel type
- Establish baseline dataset

### **Week 2: Trend and Geographic Data**
- Add Annual and Monthly operations
- Include State Specific data
- Build comprehensive view

### **Week 3: Capacity and Demand**
- Add Generator Inventory
- Include Sales data if time permits
- Complete dataset

### **Week 4: Validation and Integration**
- Cross-validate with PJM data
- Ensure temporal alignment
- Finalize dataset

---

## 💡 **KEY INSIGHTS**

1. **Daily and Hourly operations** is your most valuable dataset - prioritize this
2. **Annual and Monthly operations** provides essential seasonal context
3. **State Specific data** is crucial for PJM regional analysis
4. **Generator Inventory** helps understand capacity constraints
5. **Sales data** is useful but less critical for wholesale price forecasting

**Focus on the first 4 subcategories for a comprehensive dataset that will support robust electricity price forecasting models.**