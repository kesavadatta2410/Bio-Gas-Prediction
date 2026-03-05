# Biogas Dataset Exploratory Data Analysis Report

## Overview

This report presents the exploratory data analysis of biogas production datasets from three sources:
1. **Muscatine SCADA** - Water treatment plant sensor data
2. **DataONE AD** - Anaerobic digestion sensor data
3. **EDI SSAD** - Solid-state anaerobic digestion biogas production data

---

## MUSCATINE Datasets

### Biogas-data-dictionary-LAB-dataset

- **Rows**: 33
- **Columns**: 5
- **Memory**: 9.94 KB
- **Numeric Columns**: 0
- **Categorical Columns**: 5

**Missing Values**: 5 columns have missing data

---

### Biogas-data-dictionary-SCADA-dataset

- **Rows**: 27
- **Columns**: 5
- **Memory**: 8.64 KB
- **Numeric Columns**: 0
- **Categorical Columns**: 5

**Missing Values**: 1 columns have missing data

---

### LABS-raw

- **Rows**: 1,103
- **Columns**: 25
- **Memory**: 215.56 KB
- **Numeric Columns**: 25
- **Categorical Columns**: 0

**Missing Values**: 15 columns have missing data

**High Correlations (|r| > 0.8)**: 4 pairs found

---

### SCADA-raw

- **Rows**: 500,400
- **Columns**: 27
- **Memory**: 134873.57 KB
- **Numeric Columns**: 26
- **Categorical Columns**: 1

**Missing Values**: None

**High Correlations (|r| > 0.8)**: 5 pairs found

---

## DATAONE_AD Datasets

### sensor_ad_full_scale_ccr

- **Rows**: 309
- **Columns**: 74
- **Memory**: 193.76 KB
- **Numeric Columns**: 71
- **Categorical Columns**: 2

**Missing Values**: 65 columns have missing data

**High Correlations (|r| > 0.8)**: 208 pairs found

---

### sensor_ad_full_scale_wwc_ccr

- **Rows**: 319
- **Columns**: 34
- **Memory**: 84.86 KB
- **Numeric Columns**: 33
- **Categorical Columns**: 0

**Missing Values**: 24 columns have missing data

**High Correlations (|r| > 0.8)**: 29 pairs found

---

### sensor_ad_pilot_merged

- **Rows**: 326
- **Columns**: 48
- **Memory**: 122.38 KB
- **Numeric Columns**: 47
- **Categorical Columns**: 0

**Missing Values**: 44 columns have missing data

**High Correlations (|r| > 0.8)**: 15 pairs found

---

## EDI_SSAD Datasets

### chemical_compositions

- **Rows**: 14
- **Columns**: 5
- **Memory**: 1.45 KB
- **Numeric Columns**: 4
- **Categorical Columns**: 1

**Missing Values**: 2 columns have missing data

**High Correlations (|r| > 0.8)**: 1 pairs found

---

### energy_balance_by_particle_size

- **Rows**: 4
- **Columns**: 5
- **Memory**: 1.23 KB
- **Numeric Columns**: 1
- **Categorical Columns**: 4

**Missing Values**: None

---

### loading_conditions

- **Rows**: 21
- **Columns**: 7
- **Memory**: 2.29 KB
- **Numeric Columns**: 6
- **Categorical Columns**: 1

**Missing Values**: None

**High Correlations (|r| > 0.8)**: 3 pairs found

---

### mixture_carbon_nitrogen_ratio_bmp

- **Rows**: 8
- **Columns**: 4
- **Memory**: 0.74 KB
- **Numeric Columns**: 3
- **Categorical Columns**: 1

**Missing Values**: None

**High Correlations (|r| > 0.8)**: 3 pairs found

---

### particle_size

- **Rows**: 180
- **Columns**: 9
- **Memory**: 30.15 KB
- **Numeric Columns**: 7
- **Categorical Columns**: 2

**Missing Values**: 3 columns have missing data

**High Correlations (|r| > 0.8)**: 2 pairs found

---

### SSAD_biogas_production

- **Rows**: 735
- **Columns**: 16
- **Memory**: 124.68 KB
- **Numeric Columns**: 15
- **Categorical Columns**: 1

**Missing Values**: 12 columns have missing data

**High Correlations (|r| > 0.8)**: 4 pairs found

---

### SSAD_methane_by_TS_and_prairie_manure_ratio

- **Rows**: 24
- **Columns**: 6
- **Memory**: 2.33 KB
- **Numeric Columns**: 5
- **Categorical Columns**: 1

**Missing Values**: 3 columns have missing data

**High Correlations (|r| > 0.8)**: 1 pairs found

---

### theoretical_BMP_parameters

- **Rows**: 2
- **Columns**: 12
- **Memory**: 0.32 KB
- **Numeric Columns**: 12
- **Categorical Columns**: 0

**Missing Values**: None

**High Correlations (|r| > 0.8)**: 66 pairs found

---

### water_activity

- **Rows**: 2
- **Columns**: 8
- **Memory**: 0.34 KB
- **Numeric Columns**: 7
- **Categorical Columns**: 1

**Missing Values**: 2 columns have missing data

**High Correlations (|r| > 0.8)**: 10 pairs found

---

### liquid_reuse_biogas_production_raw

- **Rows**: 240
- **Columns**: 18
- **Memory**: 73.21 KB
- **Numeric Columns**: 14
- **Categorical Columns**: 4

**Missing Values**: 11 columns have missing data

**High Correlations (|r| > 0.8)**: 4 pairs found

---

### recirculation_biogas_production_raw

- **Rows**: 294
- **Columns**: 9
- **Memory**: 54.36 KB
- **Numeric Columns**: 7
- **Categorical Columns**: 2

**Missing Values**: 4 columns have missing data

**High Correlations (|r| > 0.8)**: 1 pairs found

---

### SSAD_biogas_production_raw

- **Rows**: 666
- **Columns**: 12
- **Memory**: 187.20 KB
- **Numeric Columns**: 8
- **Categorical Columns**: 4

**Missing Values**: 5 columns have missing data

**High Correlations (|r| > 0.8)**: 1 pairs found

---

### total_volatile_solids_raw

- **Rows**: 9
- **Columns**: 6
- **Memory**: 0.98 KB
- **Numeric Columns**: 5
- **Categorical Columns**: 1

**Missing Values**: 1 columns have missing data

**High Correlations (|r| > 0.8)**: 1 pairs found

---

## Visualizations

All visualizations have been saved to the `output/` folder, organized by data source:
- `output/muscatine/` - Muscatine SCADA dataset plots
- `output/dataone_ad/` - DataONE AD dataset plots
- `output/edi_ssad/` - EDI SSAD dataset plots

Each dataset has the following visualizations:
- Overview plot (data types, missing values, statistics)
- Correlation heatmap
- Missing values chart
- Box plots (normalized)
- Time series plots (where applicable)
