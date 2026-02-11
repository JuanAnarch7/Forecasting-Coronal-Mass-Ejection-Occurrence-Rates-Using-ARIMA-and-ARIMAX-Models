# Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models
This repository contains the implementation of ARIMA (AutoRegressive Integrated Moving Average) and ARIMAX (ARIMA with eXogenous variables) models for forecasting Coronal Mass Ejection (CME) occurrence rates.

This repository contains the implementation of ARIMA (AutoRegressive Integrated Moving Average) and ARIMAX (ARIMA with eXogenous variables) models for forecasting Coronal Mass Ejection (CME) occurrence rates. The project evaluates whether sunspot numbers (SSN) can improve CME forecasting accuracy compared to univariate time series models.
The analysis includes:

(*) Proper train/test validation (80/20 split) with out-of-sample forecasting
(*) Annual and monthly temporal resolutions using ARIMA/ARIMAX and SARIMA/SARIMAX
(*) Multiple CME subpopulations (All CMEs, Fast, Halo, Partial Halo, Fast Partial Halo)
(*) Comprehensive statistical tests (ADF, Ljung-Box, correlation analysis)
(*) Figures and detailed performance metrics

## Objectives

Compare forecasting performance of ARIMA vs. ARIMAX models for CME occurrence rates
Evaluate the predictive value of sunspot numbers for different CME subpopulations
Assess model performance using rigorous out-of-sample validation metrics
Provide publication-ready results with comprehensive statistical diagnostics


## Data Description

The analysis combines CME and sunspot datasets:

- **Sunspot numbers:** International Sunspot Number provided by SILSO (Royal Observatory of Belgium).
- **CME data:** Processed CME catalog derived from SOHO/LASCO observations.

Processed datasets are used to ensure consistent event filtering and temporal aggregation. Raw datasets are not distributed in this repository; instructions and sources are described in `data/README.md`.

---


## Repository Structure

CME_Forecasting_Repository/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory (not tracked by git)
│   ├── raw/                           # Raw data files
│   │   ├── datos_30_11_2025.txt       # Raw CME data from CDAW
│   │   ├── SN_y_tot_V2.0.txt          # Annual sunspot data from SILSO
│   │   └── SN_m_tot_V2.0.txt          # Monthly sunspot data from SILSO
│   │
│   └── processed/                     # Processed data
│       └── datos_procesados_2025_11_30.csv  # Cleaned CME data
│
├── data_processing/                   # Data preprocessing scripts
│   ├── __init__.py
│   ├── Lecture_data_CME.py            # Convert raw CME data to CSV
│   └── README.md                      # Data processing documentation
│
├── scripts/                           # Main analysis scripts
│   ├── __init__.py
│   ├── annual_forecasting.py          # ARIMA vs ARIMAX (annual)
│   ├── monthly_forecasting.py         # SARIMA vs SARIMAX (monthly)
│   └── config.py                      # Shared configuration parameters
│
├── results/                           # Output directory (not tracked by git)
│   ├── figures/                       # Generated plots
│   │   ├── acf_pacf_*.pdf
│   │   ├── forecast_comparison_*.pdf
│   │   └── residual_diagnostics_*.pdf
│   │
│   ├── tables/                        # Results tables (CSV)
│   │   └── forecast_results_*.csv
│   │
│   └── metrics/                       # Performance metrics
│       └── model_metrics_summary_*.txt

## Scripts Description

The `scripts` directory contains the main forecasting pipelines used to generate the results presented in this project:

- `annual_forecasting.py`  
  Implements ARIMA vs ARIMAX forecasting at **annual resolution**. The script filters CME events, constructs yearly occurrence rates, merges them with annual sunspot numbers, performs stationarity and autocorrelation tests, and evaluates forecasting performance using proper out-of-sample validation.

- `monthly_forecasting.py`  
  Implements SARIMA vs SARIMAX forecasting at **monthly resolution**, including seasonal components. It generates monthly CME occurrence rates, performs correlation and statistical diagnostics, fits seasonal time-series models, and produces publication-quality forecast comparisons and residual analyses.


## Quick Start

1. Clone repository
   git clone https://github.com/USER/REPO.git

2. Install dependencies
   pip install -r requirements.txt

3. Download sunspot dataset from SILSO and place it in the project folder.

4. Run desired analysis script, for example:
   python scripts/annual_forecasting.py

Analysis performed using Python 3.10.12

Data last updated: 2025-11-30

## Workflow Overview

The analysis pipeline consists of three stages:

1. Data preprocessing

Raw CME catalogs are parsed and converted into structured CSV files.

Numerical columns are cleaned and formatted.


2. Time-series construction

CME events are filtered by speed and angular width.

Monthly or annual CME occurrence rates are computed.

Sunspot datasets are merged with CME counts.

3. Forecasting and validation

ARIMA/SARIMA models are trained.

ARIMAX/SARIMAX models incorporate sunspot numbers.

Out-of-sample validation is performed.

Metrics and residual diagnostics are generated.

How to Run the Analysis
Step 1 – Preprocess CME data
python data_processing/Lecture_data_CME.py


This generates the processed CME dataset used by forecasting scripts.

Step 2 – Monthly forecasting
python scripts/monthly_forecasting.py


Runs SARIMA vs SARIMAX comparison at monthly resolution.

Step 3 – Annual forecasting
python scripts/annual_forecasting.py



    
    
