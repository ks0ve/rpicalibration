### Usage: 
To run this yourself on different data, change variables accordingly in the get_user_inputs function. Make sure you download the matching ubibot data.

### rpi_calibration.py:

##### Functionality: 
1. Parses, cleans, and converts the Ubibot and Raspberry Pi datasets to UTC.
2. Cleans the Ubibot and Raspberry Pi data
- removes nulls
- removes temperatures over 100 and below -50 celcius
- removes humidity readings over 100% and below 0%
- removes light readings below 0
3. Aggregates and merges the sensor data per minute
4. Performs calibration using linear regression, XGBoost, and random forest. 
- No hyperparameter tuning yet
- Light is not calibrated.
5. Computes R^2, MSE, RMSE, MAE, and explained variance error metrics
6. Creates visualizations
- Summary statistics table
- Histograms with overlays
- Scatterplots
- Time series plots that compare the raspberry pi feature, the ubibot feature, and the calibrated feature
- correlation matrix
- Aggregated visualizations

##### The outputs for this script are:
- Clean CSVs for GS1, WS1, RPI (gs1_clean.csv, ws1_clean.csv, rpi_clean.csv)
- Fused dataset CSV (fused_data.csv)
- PDF file containing all visualizations
- A text file of calibration equations and error metrics (calibration_equations.txt)
- Various informative console outputs