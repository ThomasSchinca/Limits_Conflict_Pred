# Replication files for Endogenous Conflict and the Limits of Predictive Optimization paper 
This repository contains the replication data for the paper "Endogenous Conflict and the Limits of Predictive" 

## Overview
The repository includes the necessary scripts and data to replicate the results presented in the paper. The primary script, `main.py`, executes the models and generates output files that visualize and store prediction and visualization results.

## Requirements
- **Python version:** 3.8.5
- Required libraries: sklearn, scipy, matplotlib and seaborn

## Running the Model
To reproduce the results, execute the following command in your terminal:
```bash
python main.py
```
## Expected Runtime
The script should take approximately 1 minute to complete.

## Directory Structure
- main.py: Main script.
- creation_data.py: Data preprocessing and storing in Data/.
- acled_test: Replication of the test with ACLED data. Need to download the data from [here](https://data.humdata.org/dataset/political-violence-events-and-fatalities), as GitHub cannot store such big dataset.
- main_horizon: Replication of the test with different horizons (t+3 and t+6) 
- Data/: Contains input data required.
- results/: Contains data produced by the model and images for visualization.

