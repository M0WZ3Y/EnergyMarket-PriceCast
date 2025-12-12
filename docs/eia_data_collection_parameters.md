# EIA Data Collection Parameters for Electricity Price Forecasting

## 🎯 **Overview**

This document identifies the essential EIA (U.S. Energy Information Administration) parameters from [eia.gov/opendata](https://eia.gov/opendata) that are critical for your electricity price forecasting thesis.

## 📊 **EIA DATA CATEGORIES FOR PRICE FORECASTING**

### **🔴 CRITICAL PRIORITY (Core Exogenous Variables)**

#### **1. Fuel Prices (Primary Cost Drivers)**
**Why Critical**: Fuel costs are major drivers of electricity prices, especially for natural gas plants.

**Key Parameters to Look For**:
- **Natural Gas Prices**
  - `NG.N3010US3` - Henry Hub Natural Gas Spot Price ($/MMBtu)
  - `NG.N3045US3` - Natural Gas Futures Prices
  - `NG.NGA_EPG0_VG_SUT_D_` - Natural Gas Price by State
  - `NG.NG_RNGR_D_` - Natural Gas Residential Prices
  - `NG.NG_RNGW_D_` - Natural Gas Wholesale Prices

- **Coal Prices**
  - `COAL.COST_APP` - Appalachian Coal Prices
  - `COAL.COST_ILB` - Illinois Basin Coal Prices
  - `COAL.COST_PRB` - Powder River Basin Coal Prices
  - `COAL.COST_US` - Average U.S. Coal Prices

- **Petroleum Products**
  - `PET.EER_EPD2D_PTE_NUS_DPG` - No. 2 Diesel Fuel Prices
  - `PET.EER_EPD2F_PTE_NUS_DPG` - No. 2 Fuel Oil Prices
  - `PET.EPP_PT_EEX_NUS_DPG` - Propane Prices

#### **2. Renewable Energy Generation**
**Why Critical**: Renewable generation impacts supply dynamics and price volatility.

**Key Parameters to Look For**:
- **Solar Generation**
  - `ELEC.GEN.SOLAR-UT-99.A` - Solar Generation by State
  - `ELEC.GEN.SOLAR-UT-99.M` - Monthly Solar Generation
  - `ELEC.GEN.SOLAR-UT-99.Y` - Annual Solar Generation

- **Wind Generation**
  - `ELEC.GEN.WND-UT-99.A` - Wind Generation by State
  - `ELEC.GEN.WND-UT-99.M` - Monthly Wind Generation
  - `ELEC.GEN.WND-UT-99.Y` - Annual Wind Generation

- **Total Renewable Generation**
  - `ELEC.GEN.REV-UT-99.A` - Total Renewable Generation
  - `ELEC.GEN.HYC-UT-99.A` - Hydroelectric Generation
  - `ELEC.GEN.GEO-UT-99.A` - Geothermal Generation
  - `ELEC.GEN.WOD-UT-99.A` - Wood and Wood Waste Generation

### **🔶 HIGH PRIORITY (Market Dynamics)**

#### **3. Electricity Market Data**
**Why Important**: Provides broader market context and regional comparisons.

**Key Parameters to Look For**:
- **Electricity Prices**
  - `ELEC.PRICE.US-RES.A` - Residential Electricity Prices
  - `ELEC.PRICE.US-COM.A` - Commercial Electricity Prices
  - `ELEC.PRICE.US-IND.A` - Industrial Electricity Prices
  - `ELEC.PRICE.US-ALL.A` - All Sectors Electricity Prices

- **Regional Electricity Prices**
  - `ELEC.PRICE.{region}-RES.A` - Regional Residential Prices
  - `ELEC.PRICE.{region}-COM.A` - Regional Commercial Prices
  - `ELEC.PRICE.{region}-IND.A` - Regional Industrial Prices

#### **4. Generation Capacity**
**Why Important**: Capacity constraints affect price formation.

**Key Parameters to Look For**:
- **Total Generation Capacity**
  - `ELEC.CAP.TOT-UT-99.A` - Total Capacity by State
  - `ELEC.CAP.NG-UT-99.A` - Natural Gas Capacity
  - `ELEC.CAP.COAL-UT-99.A` - Coal Capacity
  - `ELEC.CAP.NUC-UT-99.A` - Nuclear Capacity

- **Renewable Capacity**
  - `ELEC.CAP.SOLAR-UT-99.A` - Solar Capacity
  - `ELEC.CAP.WND-UT-99.A` - Wind Capacity
  - `ELEC.CAP.REV-UT-99.A` - Total Renewable Capacity

### **🔹 MEDIUM PRIORITY (Advanced Features)**

#### **5. Energy Consumption**
**Why Useful**: Demand-side indicators for price forecasting.

**Key Parameters to Look For**:
- **Electricity Consumption**
  - `ELEC.CONS.US-RES.A` - Residential Consumption
  - `ELEC.CONS.US-COM.A` - Commercial Consumption
  - `ELEC.CONS.US-IND.A` - Industrial Consumption
  - `ELEC.CONS.US-TOT.A` - Total Consumption

- **Regional Consumption**
  - `ELEC.CONS.{region}-TOT.A` - Regional Total Consumption

#### **6. Fuel Consumption**
**Why Useful**: Fuel demand indicators affecting prices.

**Key Parameters to Look For**:
- **Natural Gas Consumption**
  - `NG.N9140US2` - Total Natural Gas Consumption
  - `NG.N9142US2` - Natural Gas Consumption by Sector

- **Coal Consumption**
  - `COAL.CONSUM_US` - Total Coal Consumption
  - `COAL.CONSUM_ELEC` - Coal Consumption for Electricity

---

## 🗺️ **GEOGRAPHIC FOCUS FOR PJM REGION**

### **Primary States in PJM Territory**
Focus on these states for regional data:
- **Pennsylvania (PA)**
- **New Jersey (NJ)**
- **Maryland (MD)**
- **Delaware (DE)**
- **Virginia (VA)**
- **West Virginia (WV)**
- **Ohio (OH)**
- **North Carolina (NC)**
- **District of Columbia (DC)**

### **Regional Codes**
- **Mid-Atlantic**: `MIDA`
- **Midwest**: `MIDW`
- **Northeast**: `NE`
- **Southeast**: `SE`
- **Texas**: `TEX`
- **California**: `CAL`
- **Florida**: `FLA`
- **Central**: `CENT`

---

## 📋 **EIA API IMPLEMENTATION STRATEGY**

### **API Endpoint Structure**
```
https://api.eia.gov/v2/
```

### **Key API Parameters**
- **api_key**: Your EIA API key
- **frequency**: `hourly`, `daily`, `weekly`, `monthly`, `annual`
- **data**: `value`, `price`, `generation`, `consumption`
- **facets**: Geographic and fuel type filters
- **start**: Start date (YYYY-MM-DD)
- **end**: End date (YYYY-MM-DD)
- **sort**: Sort by period and direction
- **offset**: Pagination offset
- **length**: Number of records per request

### **Example API Calls**
```python
# Natural Gas Prices
https://api.eia.gov/v2/ng/data/?frequency=daily&data[0]=value&facets[series][]=NG.N3010US3&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY

# Solar Generation
https://api.eia.gov/v2/electricity/generation/data/?frequency=monthly&data[0]=generation&facets[fueltypeid][]=SUN&facets[stateid][]=PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY

# Wind Generation
https://api.eia.gov/v2/electricity/generation/data/?frequency=monthly&data[0]=generation&facets[fueltypeid][]=WND&facets[stateid][]=PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

---

## 🎯 **PRIORITY IMPLEMENTATION ORDER**

### **Phase 1: Core Fuel Prices (Week 1)**
1. **Natural Gas Prices** - Henry Hub and regional prices
2. **Coal Prices** - Major coal basin prices
3. **Petroleum Prices** - No. 2 fuel oil and diesel

### **Phase 2: Renewable Generation (Week 2)**
4. **Solar Generation** - PJM states monthly data
5. **Wind Generation** - PJM states monthly data
6. **Total Renewable Generation** - All renewable sources

### **Phase 3: Market Context (Week 3)**
7. **Electricity Prices** - Regional price comparisons
8. **Generation Capacity** - Capacity by fuel type
9. **Energy Consumption** - Demand-side indicators

### **Phase 4: Advanced Features (Week 4)**
10. **Fuel Consumption** - Demand indicators
11. **Regional Analysis** - State-level breakdowns
12. **Historical Trends** - Long-term patterns

---

## 📊 **DATA QUALITY REQUIREMENTS**

### **Temporal Coverage**
- **Historical Data**: 2014-2024 (10 years)
- **Update Frequency**: Daily for prices, monthly for generation
- **Consistency**: Continuous time series without gaps

### **Geographic Coverage**
- **Primary Focus**: PJM territory states
- **Secondary**: Major U.S. regions for comparison
- **Resolution**: State-level primary, regional secondary

### **Data Validation**
- **Cross-Validation**: Compare with PJM fuel data
- **Outlier Detection**: Identify unusual price spikes
- **Missing Data**: Implement interpolation strategies

---

## 🔧 **TECHNICAL IMPLEMENTATION**

### **Data Storage Structure**
```
02_data/raw/fuel_renewables/eia/
├── fuel_prices/
│   ├── natural_gas/
│   ├── coal/
│   └── petroleum/
├── renewable_generation/
│   ├── solar/
│   ├── wind/
│   └── total_renewable/
├── electricity_prices/
├── generation_capacity/
└── energy_consumption/
```

### **Processing Pipeline**
1. **API Data Collection** - Automated daily/weekly collection
2. **Data Validation** - Quality checks and outlier detection
3. **Data Transformation** - Standardize formats and units
4. **Feature Engineering** - Create price ratios and trends
5. **Storage** - Save in Parquet format with compression

---

## 🚀 **NEXT STEPS**

1. **Get EIA API Key**: Register at [eia.gov/opendata](https://eia.gov/opendata)
2. **Test API Access**: Verify connectivity and data availability
3. **Implement Collector**: Build EIA data collection module
4. **Set Up Scheduling**: Automated data collection
5. **Validate Data**: Cross-check with PJM data

This EIA data collection strategy provides the essential exogenous variables needed to complement your PJM data for comprehensive electricity price forecasting.