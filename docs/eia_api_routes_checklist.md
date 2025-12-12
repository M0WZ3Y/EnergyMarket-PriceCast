# EIA API Routes Collection Checklist

## 🎯 **EIA API ROUTES PRIORITY CHECKLIST**

### **STATUS LEGEND**
- 🔴 **NOT STARTED** - Route not yet accessed
- 🟡 **IN PROGRESS** - Currently exploring route
- 🟢 **COMPLETED** - Route successfully accessed and data collected
- ❌ **SKIPPED** - Route deemed not relevant for thesis
- ⚠️ **ERROR** - Route access failed or data unavailable

---

## 🔴 **PHASE 1: CRITICAL ROUTES (Must Complete First)**

### **1. Electricity** 🟡 **IN PROGRESS**
**Priority**: 🔴 **CRITICAL** | **Status**: 🟡 **IN PROGRESS**
**Why Critical**: Core electricity market data, generation, consumption, prices

**Available Subcategories**:
- [ ] **Balancing Authority Areas hourly operating data** ⭐ **MOST IMPORTANT**
  - [ ] **Electric Power Operations (Daily and Hourly)** ⭐ **PRIMARY TARGET**
    - [ ] **Hourly Demand, Demand Forecast, Generation, and Interchange** 🔴 **CRITICAL**
    - [ ] **Hourly Generation by Energy Source** 🔴 **CRITICAL**
    - [ ] **Hourly Demand by Subregion** 🟡 **HIGH PRIORITY**
    - [ ] **Hourly Interchange by Neighboring Balancing Authority** 🟡 **HIGH PRIORITY**
    - [ ] **Daily Demand, Demand Forecast, Generation, and Interchange** 🟡 **HIGH PRIORITY**
    - [ ] **Daily Demand by Subregion** 🔹 **MEDIUM PRIORITY**
    - [ ] **Daily Generation by Energy Source** 🟡 **HIGH PRIORITY**
    - [ ] **Daily Interchange by Neighboring Balancing Authority** 🔹 **MEDIUM PRIORITY**
  - [ ] **Electric Power Operations (Annual and Monthly)** 🔹 **MEDIUM PRIORITY**
- [x] **Balancing Authority Areas hourly operating data** ✅ **COMPLETED**
  - [x] **Electric Power Operations (Daily and Hourly)** ✅ **COMPLETED**
    - [x] **Hourly Demand, Demand Forecast, Generation, and Interchange** ✅ **COMPLETED**
    - [x] **Hourly Generation by Energy Source** ✅ **COMPLETED**
    - [x] **Hourly Demand by Subregion** ✅ **COMPLETED**
    - [x] **Hourly Interchange by Neighboring Balancing Authority** ✅ **COMPLETED**
    - [x] **Daily Demand, Demand Forecast, Generation, and Interchange** ✅ **COMPLETED**
    - [x] **Daily Demand by Subregion** ✅ **COMPLETED**
    - [x] **Daily Generation by Energy Source** ✅ **COMPLETED**
    - [x] **Daily Interchange by Neighboring Balancing Authority** ✅ **COMPLETED**
  - [x] **Electric Power Operations (Annual and Monthly)** ✅ **COMPLETED**
- [🔄] **Generation** 🟡 **CURRENTLY EXPLORING**
  - [🔄] **Monthly generation by fuel type, sector, and state** 🔴 **CRITICAL**
  - [🔄] **Monthly generation by plant, fuel type and prime mover** 🔴 **CRITICAL**
- [ ] **Capability** 🟡 **HIGH PRIORITY**
  - Monthly generator-level capability by fuel type, sector, and state
- [ ] **Retail sales** 🟡 **HIGH PRIORITY**
  - Monthly price, customers, revenue, and sales by sector and state
- [ ] **State-level electricity data** 🔹 **MEDIUM PRIORITY**
  - Annually by state and fuel

**Data to Collect**:
- [x] **Hourly Demand, Demand Forecast, Generation, and Interchange** ✅ **COMPLETED**
- [x] **Hourly Generation by Energy Source** ✅ **COMPLETED** (NG, COAL, SUN, WND)
- [x] **Hourly Demand by Subregion** ✅ **COMPLETED** (PJM subregions)
- [x] **Hourly Interchange by Neighboring Balancing Authority** ✅ **COMPLETED** (PJM imports/exports)
- [x] **Daily Generation by Energy Source** ✅ **COMPLETED** (Daily fuel mix)
- [🔄] **Monthly generation by fuel type, sector, and state** 🔴 **CURRENT TARGET**
- [🔄] **Monthly generation by plant, fuel type and prime mover** 🔴 **CURRENT TARGET**
- [ ] **Generator capability data** (Capability) 🟡 **HIGH**
- [ ] **Retail prices and sales** (Retail sales) 🟡 **HIGH**
- [ ] **Annual state-level data** (State-level) 🔹 **MEDIUM**

**API Endpoints to Explore**:
```
/electricity/balancing-authority/data/
/electricity/generation/data/
/electricity/capability/data/
/electricity/retail-sales/data/
/electricity/state-level-data/data/
```

---

### **2. Natural Gas** 🔴 **NOT STARTED**
**Priority**: 🔴 **CRITICAL** | **Status**: 🔴 **NOT STARTED**
**Why Critical**: Primary fuel for electricity generation in PJM region

**Data to Collect**:
- [ ] Henry Hub natural gas spot prices
- [ ] Natural gas prices by state
- [ ] Natural gas consumption by sector
- [ ] Natural gas storage levels
- [ ] Natural gas production data

**API Endpoints to Explore**:
```
/natural-gas/pri/fut/data/
/natural-gas/sum/nwly/sum/data/
/natural-gas/stor/sum/data/
/natural-gas/consumption/data/
```

---

### **3. Coal** 🔴 **NOT STARTED**
**Priority**: 🔴 **CRITICAL** | **Status**: 🔴 **NOT STARTED**
**Why Critical**: Significant generation fuel in PJM region

**Data to Collect**:
- [ ] Coal prices by region (Appalachian, Illinois Basin, PRB)
- [ ] Coal consumption by sector
- [ ] Coal production and inventories
- [ ] Coal exports and imports

**API Endpoints to Explore**:
```
/coal/prices/data/
/coal/consumption/data/
/coal/production/data/
/coal/stocks/data/
```

---

## 🟡 **PHASE 2: HIGH PRIORITY ROUTES**

### **4. Petroleum** 🟡 **NOT STARTED**
**Priority**: 🟡 **HIGH** | **Status**: 🟡 **NOT STARTED**
**Why Important**: Petroleum products impact electricity generation costs

**Data to Collect**:
- [ ] No. 2 fuel oil prices
- [ ] Diesel fuel prices
- [ ] Petroleum consumption by sector
- [ ] Petroleum inventories

**API Endpoints to Explore**:
```
/petroleum/pri/gnd/data/
/petroleum/cons/sum/data/
/petroleum/stoc/sum/data/
```

---

### **5. State Energy Data System (SEDS)** 🟡 **NOT STARTED**
**Priority**: 🟡 **HIGH** | **Status**: 🟡 **NOT STARTED**
**Why Important**: Comprehensive state-level energy data for PJM territories

**Data to Collect**:
- [ ] State-level energy consumption
- [ ] State-level energy production
- [ ] State-level energy prices
- [ ] State-level renewable energy data

**API Endpoints to Explore**:
```
/seds/data/
/seds/sum/data/
/seds/rl/data/
```

---

## 🔹 **PHASE 3: MEDIUM PRIORITY ROUTES**

### **6. Outlook of Energy Market/Projections Data** 🔹 **NOT STARTED**
**Priority**: 🔹 **MEDIUM** | **Status**: 🔹 **NOT STARTED**
**Why Useful**: Forward-looking energy price and consumption forecasts

**Data to Collect**:
- [ ] Short-term electricity price forecasts
- [ ] Short-term natural gas price forecasts
- [ ] Energy consumption forecasts
- [ ] Renewable energy forecasts

**API Endpoints to Explore**:
```
/steo/data/
/steo/sum/data/
```

---

### **7. Total Energy** 🔹 **NOT STARTED**
**Priority**: 🔹 **MEDIUM** | **Status**: 🔹 **NOT STARTED**
**Why Useful**: Comprehensive energy overview and trends

**Data to Collect**:
- [ ] Total energy consumption
- [ ] Energy intensity metrics
- [ ] Renewable energy share
- [ ] Energy price indices

**API Endpoints to Explore**:
```
/total-energy/data/
/total-energy/sum/data/
```

---

## ❌ **PHASE 4: LOW PRIORITY/SKIP ROUTES**

### **8. Crude Oil Imports** ❌ **SKIP**
**Priority**: ❌ **LOW** | **Status**: ❌ **SKIP**
**Why Skip**: Limited impact on U.S. electricity prices

**Reason**: Crude oil imports primarily affect transportation sector, not electricity generation in PJM region.

---

### **9. Densified Biomass** ❌ **SKIP**
**Priority**: ❌ **LOW** | **Status**: ❌ **SKIP**
**Why Skip**: Niche renewable data, minimal impact on PJM prices

**Reason**: Biomass represents small fraction of PJM generation mix.

---

### **10. Nuclear Plant Generator Outages** ❌ **SKIP**
**Priority**: ❌ **LOW** | **Status**: ❌ **SKIP**
**Why Skip**: Specialized data, limited forecasting value

**Reason**: Nuclear outages are relatively predictable and already factored into PJM operational data.

---

### **11. CO2 Emissions** ❌ **SKIP**
**Priority**: ❌ **LOW** | **Status**: ❌ **SKIP**
**Why Skip**: Environmental data, limited price forecasting value

**Reason**: CO2 emissions are outcome data, not input for price forecasting.

---

### **12. International Energy** ❌ **SKIP**
**Priority**: ❌ **LOW** | **Status**: ❌ **SKIP**
**Why Skip**: International focus, not PJM-specific

**Reason**: International energy markets have limited direct impact on PJM electricity prices.

---

## 📊 **COLLECTION PROGRESS TRACKER**

### **Overall Progress**
- **Total Routes**: 12
- **Critical Routes**: 3 (Electricity, Natural Gas, Coal)
- **High Priority**: 2 (Petroleum, SEDS)
- **Medium Priority**: 2 (Outlook, Total Energy)
- **Skip Routes**: 5

### **Current Status**
- **Completed**: 0/12 (0%)
- **In Progress**: 0/12 (0%)
- **Not Started**: 7/12 (58%)
- **Skipped**: 5/12 (42%)

---

## 🚀 **CURRENT ACTION PLAN - ELECTRICITY ROUTE**

### **🔴 STEP 1: Balancing Authority Areas hourly operating data** ⭐ **CURRENT**
**Status**: 🟡 **EXPLORING NOW**
**Why Critical**: Hourly demand and generation data - perfect for price forecasting
**Data Needed**:
- [ ] Actual demand (hourly)
- [ ] Forecast demand (hourly)
- [ ] Net generation (hourly)
- [ ] Power flow between systems (hourly)
**Geographic Focus**: Look for PJM Balancing Authority data
**Time Period**: 2014-2024

### **🔴 STEP 2: Generation Subcategory**
**Status**: 🔴 **NEXT PRIORITY**
**Why Critical**: Monthly generation by fuel type and plant-level data
**Data Needed**:
- [ ] Monthly generation by fuel type (NG, COAL, SUN, WND)
- [ ] Monthly generation by plant
- [ ] Sector-specific generation data
- [ ] State-level generation breakdown

### **🟡 STEP 3: Capability Subcategory**
**Status**: 🟡 **HIGH PRIORITY**
**Why Important**: Generator capacity constraints
**Data Needed**:
- [ ] Monthly generator-level capability
- [ ] Capability by fuel type
- [ ] Sector and state breakdown

### **🟡 STEP 4: Retail Sales Subcategory**
**Status**: 🟡 **HIGH PRIORITY**
**Why Important**: Retail price context and demand patterns
**Data Needed**:
- [ ] Monthly prices by sector
- [ ] Customer counts
- [ ] Revenue data
- [ ] Sales volumes

### **🔹 STEP 5: State-level Electricity Data**
**Status**: 🔹 **MEDIUM PRIORITY**
**Why Useful**: Annual state-level summaries
**Data Needed**:
- [ ] Annual data by state
- [ ] Fuel type breakdowns

---

## 🎯 **NEXT ROUTES AFTER ELECTRICITY**

### **Step 6: Natural Gas Route**
**Action**: After completing Electricity subcategories
**Focus**: Henry Hub prices and state-level data

### **Step 7: Coal Route**
**Action**: After Natural Gas
**Focus**: Regional coal prices and consumption

### **Step 8: Petroleum Route**
**Action**: After Coal
**Focus**: Fuel oil and diesel prices

---

## 📋 **DATA COLLECTION CHECKLIST**

### **For Each Route**:
- [ ] Access API route successfully
- [ ] Browse available datasets
- [ ] Check data coverage (2014-2024)
- [ ] Verify geographic coverage (PJM states)
- [ ] Test API endpoints with sample calls
- [ ] Document available parameters
- [ ] Note any data limitations
- [ ] Update route status

### **Quality Requirements**:
- [ ] Temporal coverage: 2014-2024
- [ ] Geographic coverage: PJM states
- [ ] Data completeness: >95%
- [ ] Update frequency: Daily/Monthly
- [ ] API reliability: Stable access

---

## 🔄 **STATUS UPDATE LOG**

### **[Date]: [Time] - Initial Setup**
- Created checklist with 12 EIA routes
- Prioritized routes for electricity price forecasting
- Identified 3 critical, 2 high priority, 2 medium priority routes
- Marked 5 routes as skip

### **[Waiting for your first route access...]**

---

## 💡 **NEXT STEPS**

1. **Start with Electricity route** - Most critical for your thesis
2. **Focus on Daily and Hourly Operations** - Primary dataset needed
3. **Document findings** - Update checklist as you progress
4. **Move to Natural Gas** - Second priority after Electricity
5. **Complete Coal** - Third critical route

**Ready to begin! Start with the Electricity API route and I'll update the checklist as you progress.**