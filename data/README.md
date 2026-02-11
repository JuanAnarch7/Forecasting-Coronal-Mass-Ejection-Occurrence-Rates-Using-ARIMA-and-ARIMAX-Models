# Data Directory

This folder documents the datasets used in the CME annual and monthly forecasting

## Included Dataset

### Processed CME Catalog

File: "datos_procesados_2025_11_30.csv"

This file contains CME events derived from the SOHO/LASCO CME
catalog after preprocessing and its obtained with the execution of Lecture.py script.


#### Script Function

The script:

- Reads the raw CME catalog file
- Extracts relevant columns
- Converts dates to standard datetime format
- Removes invalid or incomplete entries
- Converts numeric columns properly
- Exports a clean CSV dataset

The dataset is lightweight (~3 MB) and is included to allow direct
reproducibility of the analysis.

### Column Language

Column names are kept in Spanish to match the original processing
pipeline:

- `Fecha` — CME occurrence date
- `Rapidez` — CME linear speed (km/s)
- `Ancho` — angular width (degrees)
- `Central` — central position angle (degrees)

## CME Data Source

Raw CME observations come from:

SOHO/LASCO CME Catalog  
https://cdaw.gsfc.nasa.gov/CME_list/

## CME Preprocessing Steps

The raw catalog was processed before analysis:

1. Conversion of event dates to standard datetime format.
2. Removal of events with missing speed or invalid entries.
3. Conversion of numerical columns to numeric format.
4. Generation of csv file with precessed data.

These steps allow direct use in statistical analyses without further cleaning.

## Sunspot Data

Sunspot numbers are **not included** in this repository and should
be downloaded from:

SILSO – Royal Observatory of Belgium  
https://www.sidc.be/silso/

The scripts automatically read the downloaded dataset once placed in
the project directory.

## Reproducibility Notes

Processed CME data is included to ensure immediate reproducibility.

Raw data can be reprocessed using the provided script.

Forecasting scripts rely only on processed datasets.

Users may regenerate processed files using updated catalogs.
