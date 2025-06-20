import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#
# Read data in with timestamp as index
#

ws1 = pd.read_csv('data/645/gs1est.csv', parse_dates=['created_at'], date_parser = lambda x: pd.to_datetime(x))
gs1 = pd.read_csv('data/645/ws1est.csv', parse_dates=['created_at'], date_parser = lambda x: pd.to_datetime(x))

#
# Keep relevant cols
#
ws1 = ws1.iloc[:,:3].dropna()
gs1 = gs1.iloc[:,:3].dropna()

#
# See dataframe info 
#

ws1.info()
gs1.info()

#
# Normalize timestamps using minute floor
#

for df in (ws1, gs1):
    df['created_at'] = df['created_at'].dt.floor('min')

#
# Rename cols to avoid confusion
#

ws1.rename(columns={"field1(Temperature ºC )" : "temp_ws1", "field2(Humidity)" : "humidity_ws1"}, inplace=True)
gs1.rename(columns={"field1(Temperature ºC )" : "temp_gs1", "field2(Humidity)" : "humidity_gs1"}, inplace=True)
df = pd.merge(gs1, ws1, on=['created_at'], how='inner')
df.set_index('created_at', inplace=True)

#
# Calculate offsets
#

avg_temp_offset = (df['temp_gs1'] - df['temp_ws1']).mean()
avg_humidity_offset = (df['humidity_gs1'] - df['humidity_ws1']).mean()
print(f"Average temperature offset: {avg_temp_offset:.2f}\nAverage humidity offset: {avg_humidity_offset:.2f}")

#
# Apply offsets
#

df['temp_ws1_calibrated'] = df['temp_ws1'] + avg_temp_offset
df['humidity_ws1_calibrated'] = df['humidity_ws1'] + avg_humidity_offset


### Linreg ###

from sklearn.linear_model import LinearRegression
import numpy as np

# Fit linear model: WS1 → GS1 for temperature
X_temp = df['temp_ws1'].values.reshape(-1, 1)
y_temp = df['temp_gs1'].values
temp_model = LinearRegression().fit(X_temp, y_temp)

# Apply model to correct WS1 temperature
df['temp_ws1_calibrated'] = temp_model.predict(X_temp)

# Repeat for humidity
X_hum = df['humidity_ws1'].values.reshape(-1, 1)
y_hum = df['humidity_gs1'].values
hum_model = LinearRegression().fit(X_hum, y_hum)

df['humidity_ws1_calibrated'] = hum_model.predict(X_hum)

# Print model coefficients
print(f"Temperature model: y = {temp_model.coef_[0]:.4f} * x + {temp_model.intercept_:.4f}")
print(f"Humidity model: y = {hum_model.coef_[0]:.4f} * x + {hum_model.intercept_:.4f}")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Create PDF file to save plots
with PdfPages("calibration_plots.pdf") as pdf:

    # Temperature before calibration
    plt.figure()
    sns.lineplot(data=df[['temp_ws1', 'temp_gs1']])
    plt.legend(labels=['WS1 (Raw)', 'GS1'])
    plt.title('Temperature Before Calibration')
    pdf.savefig()  # Save current figure to PDF
    plt.close()

    # Temperature after calibration
    plt.figure()
    sns.lineplot(data=df[['temp_ws1_calibrated', 'temp_gs1']])
    plt.legend(labels=['WS1 (Calibrated)', 'GS1'])
    plt.title('Temperature After Linear Calibration')
    pdf.savefig()
    plt.close()

    # Humidity before calibration
    plt.figure()
    sns.lineplot(data=df[['humidity_ws1', 'humidity_gs1']])
    plt.legend(labels=['WS1 (Raw)', 'GS1'])
    plt.title('Humidity Before Calibration')
    pdf.savefig()
    plt.close()

    # Humidity after calibration
    plt.figure()
    sns.lineplot(data=df[['humidity_ws1_calibrated', 'humidity_gs1']])
    plt.legend(labels=['WS1 (Calibrated)', 'GS1'])
    plt.title('Humidity After Linear Calibration')
    pdf.savefig()
    plt.close()
