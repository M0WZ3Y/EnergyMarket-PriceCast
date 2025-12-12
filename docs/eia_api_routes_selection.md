# EIA API Routes Selection for Electricity Price Forecasting

## 🎯 **API ROUTE SELECTION**

Based on the available EIA API routes, here are the **essential routes** you need for your electricity price forecasting thesis:

---

## 🔴 **CRITICAL PRIORITY API ROUTES**

### **1. Electricity API Route**
**Why Critical**: Direct electricity market data, generation, and consumption

**Key Data to Collect**:
- Electricity generation by fuel type
- Electricity consumption by sector
- Electricity capacity by fuel type
- Regional electricity prices
- Demand and supply metrics

**API Endpoint**: `https://api.eia.gov/v2/electricity/`

### **2. Natural Gas API Route**
**Why Critical**: Natural gas is the primary fuel for electricity generation in PJM

**Key Data to Collect**:
- Natural gas spot prices (Henry Hub)
- Natural gas prices by state
- Natural gas consumption by sector
- Natural gas storage levels
- Natural gas production

**API Endpoint**: `https://api.eia.gov/v2/natural-gas/`

### **3. Coal API Route**
**Why Critical**: Coal remains a significant generation fuel in PJM region

**Key Data to Collect**:
- Coal prices by region
- Coal consumption by sector
- Coal production and inventories
- Coal exports and imports

**API Endpoint**: `https://api.eia.gov/v2/coal/`

---

## 🔶 **HIGH PRIORITY API ROUTES**

### **4. State Energy Data System (SEDS) API Route**
**Why Important**: Comprehensive state-level energy data for PJM territories

**Key Data to Collect**:
- State-level energy consumption
- State-level energy production
- State-level energy prices
- State-level renewable energy data

**API Endpoint**: `https://api.eia.gov/v2/seds/`

### **5. Petroleum API Route**
**Why Important**: Petroleum products impact electricity generation costs

**Key Data to Collect**:
- No. 2 fuel oil prices
- Diesel fuel prices
- Petroleum consumption by sector
- Petroleum inventories

**API Endpoint**: `https://api.eia.gov/v2/petroleum/`

---

## 🔹 **MEDIUM PRIORITY API ROUTES**

### **6. Short Term Energy Outlook API Route**
**Why Useful**: Forward-looking energy price and consumption forecasts

**Key Data to Collect**:
- Short-term electricity price forecasts
- Short-term natural gas price forecasts
- Energy consumption forecasts
- Renewable energy forecasts

**API Endpoint**: `https://api.eia.gov/v2/steo/`

### **7. Total Energy API Route**
**Why Useful**: Comprehensive energy overview and trends

**Key Data to Collect**:
- Total energy consumption
- Energy intensity metrics
- Renewable energy share
- Energy price indices

**API Endpoint**: `https://api.eia.gov/v2/total-energy/`

---

## ❌ **NOT RECOMMENDED FOR THESIS**

### **Low Priority Routes**
- **Crude Oil Imports**: Limited impact on U.S. electricity prices
- **International**: Focus on international markets, less relevant for PJM
- **Nuclear Outages**: Specialized data, limited forecasting value
- **Densified Biomass**: Niche renewable data
- **Annual Energy Outlook**: Long-term forecasts, not suitable for daily/hourly forecasting
- **International Energy Outlook**: International focus, not U.S. regional
- **State CO2 Emissions**: Environmental data, limited price forecasting value

---

## 🎯 **IMPLEMENTATION STRATEGY**

### **Phase 1: Core Routes (Week 1-2)**
1. **Electricity API** - Primary electricity market data
2. **Natural Gas API** - Primary fuel cost data
3. **Coal API** - Secondary fuel cost data

### **Phase 2: Regional Analysis (Week 3)**
4. **SEDS API** - State-level detailed data
5. **Petroleum API** - Tertiary fuel cost data

### **Phase 3: Forecast Enhancement (Week 4)**
6. **Short Term Energy Outlook API** - Forward-looking data
7. **Total Energy API** - Comprehensive energy context

---

## 📊 **SPECIFIC ENDPOINTS TO EXPLORE**

### **Electricity API Endpoints**
```
/electricity/rto/region-type-data/data/          # RTO electricity data
/electricity/electric-power-operational-data/data/  # Generation data
/electricity/retail-sales/data/                  # Retail electricity prices
/electricity/power-operational-data/data/        # Operational data
```

### **Natural Gas API Endpoints**
```
/natural-gas/pri/fut/data/                       # Natural gas futures
/natural-gas/sum/nwly/sum/data/                  # Weekly natural gas data
/natural-gas/stor/sum/data/                      # Natural gas storage
/natural-gas/pri/sum/data/                       # Natural gas prices summary
```

### **Coal API Endpoints**
```
/coal/consumption/data/                          # Coal consumption
/coal/production/data/                           # Coal production
/coal/stocks/data/                               # Coal inventories
/coal/prices/data/                               # Coal prices
```

### **SEDS API Endpoints**
```
/seds/data/                                      # State energy data
/seds/sum/data/                                  # Summary data
/seds/rl/data/                                   # Renewable energy data
```

---

## 🔧 **API CALL EXAMPLES**

### **Electricity Generation Data**
```python
https://api.eia.gov/v2/electricity/electric-power-operational-data/data/?frequency=monthly&data[0]=generation&facets[fueltypeid][]=NG&facets[stateid][]=PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

### **Natural Gas Prices**
```python
https://api.eia.gov/v2/natural-gas/pri/fut/data/?frequency=daily&data[0]=value&facets[series][]=NG.N3010US3&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

### **Coal Prices**
```python
https://api.eia.gov/v2/coal/prices/data/?frequency=monthly&data[0]=value&facets[location][]=APP&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

### **SEDS State Data**
```python
https://api.eia.gov/v2/seds/data/?frequency=annual&data[0]=value&facets[stateid][]=PA&facets[seriesid][]=ETCBU.PA&start=2014-01-01&end=2024-12-31&api_key=YOUR_API_KEY
```

---

## 📋 **SELECTION CHECKLIST**

### **Must-Select Routes**
- [ ] **Electricity** - Core electricity market data
- [ ] **Natural Gas** - Primary fuel cost driver
- [ ] **Coal** - Secondary fuel cost driver

### **Should-Select Routes**
- [ ] **State Energy Data System (SEDS)** - State-level analysis
- [ ] **Petroleum** - Tertiary fuel costs

### **Optional Routes**
- [ ] **Short Term Energy Outlook** - Forecast enhancement
- [ ] **Total Energy** - Comprehensive context

### **Skip Routes**
- [ ] **Crude Oil Imports** - Limited relevance
- [ ] **International** - International focus
- [ ] **Nuclear Outages** - Specialized data
- [ ] **Densified Biomass** - Niche data
- [ ] **Annual Energy Outlook** - Long-term focus
- [ ] **International Energy Outlook** - International focus
- [ ] **State CO2 Emissions** - Environmental focus

---

## 🚀 **NEXT STEPS**

1. **Visit [eia.gov/opendata](https://eia.gov/opendata)**
2. **Navigate to these API routes**:
   - Electricity
   - Natural Gas
   - Coal
   - State Energy Data System (SEDS)
   - Petroleum

3. **For each route**:
   - Browse available datasets
   - Check data coverage (2014-2024)
   - Verify geographic coverage (PJM states)
   - Test API endpoints

4. **Priority Order**:
   1. Start with Electricity API
   2. Add Natural Gas API
   3. Include Coal API
   4. Add SEDS for state-level detail
   5. Include Petroleum for completeness

This selection provides the most relevant and impactful data for your electricity price forecasting thesis while avoiding unnecessary complexity.