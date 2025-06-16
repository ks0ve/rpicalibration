import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# 1. CREATE OUTPUT DIRECTORY AND PDF
# ------------------------------------------------
outdir = input("Enter output directory within figures folder for plots: ")
outfile = input("Enter output pdf name (e.g., analysis_plots.pdf): ")
output_dir = f"figures/{outdir}"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, outfile)
print(f"Saving figures to: {output_dir} as '{outfile}'")

# ------------------------------------------------
# 2. LOAD RAW CSVs (TIMESTAMPS PARSED)
# ------------------------------------------------
relpath    = input("Enter data folder name: ")
file_gs1   = input("Enter GS1 data filename: ")
file_ws1   = input("Enter WS1 data filename: ")
file_raspi = input("Enter RPi data filename: ")

gs1 = pd.read_csv(
    f"{relpath}/{file_gs1}",
    parse_dates=['created_at'],
    date_parser=lambda x: pd.to_datetime(x, utc=True)
)
ws1 = pd.read_csv(
    f"{relpath}/{file_ws1}",
    parse_dates=['created_at'],
    date_parser=lambda x: pd.to_datetime(x, utc=True)
)
rpi = pd.read_csv(
    f"{relpath}/{file_raspi}",
    parse_dates=['Timestamp']
)

# ------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------
# GS1: keep only timestamp, temperature, humidity
gs1_clean = gs1[['created_at','field1(Temperature ºC )','field2(Humidity)']].copy()
gs1_clean.rename(columns={
    'created_at': 'Timestamp',
    'field1(Temperature ºC )': 'Temperature (°C)',
    'field2(Humidity)':       'Humidity (%)'
}, inplace=True)

# WS1: keep only timestamp, ambient light
ws1_clean = ws1[['created_at','field3(Light)']].copy()
ws1_clean.rename(columns={
    'created_at':    'Timestamp',
    'field3(Light)': 'Ambient Light (lux)'
}, inplace=True)

# RPi: drop color band & gas-resistance, keep only the four columns
rpi_clean = rpi[['Timestamp','Temperature (°C)','Humidity (%)','Ambient Light (lux)']].copy()

# ------------------------------------------------
# 4. NORMALIZE TIMESTAMPS TO MINUTE PRECISION
# ------------------------------------------------
for df in (gs1_clean, ws1_clean):
    df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    df['minute_ts'] = df['Timestamp'].dt.floor('min')

rpi_clean['Timestamp'] = rpi_clean['Timestamp'].dt.tz_localize(None)
rpi_clean['minute_ts'] = rpi_clean['Timestamp'].dt.floor('min')

# ------------------------------------------------
# 5. AGGREGATE READINGS PER MINUTE (MEAN)
# ------------------------------------------------
def aggregate_minute(df, prefix):
    agg = {
        'Temperature (°C)': 'mean',
        'Humidity (%)': 'mean',
        'Ambient Light (lux)' : 'mean'
    }
    if prefix == 'GS1':
        agg.pop('Ambient Light (lux)')
    if prefix == 'WS1':
        agg = { 'Ambient Light (lux)': 'mean'}
    grouped = df.groupby('minute_ts').agg(agg).reset_index()
    # rename columns
    rename_map = {
        'Temperature (°C)':       f'Temperature_{prefix}',
        'Humidity (%)':           f'Humidity_{prefix}'
    }
    if prefix in ('WS1','RPI'):
        rename_map['Ambient Light (lux)'] = f'Ambient_Light_{prefix}'
    grouped.rename(columns=rename_map, inplace=True)
    return grouped

gs1_agg = aggregate_minute(gs1_clean, 'GS1')
ws1_agg = aggregate_minute(ws1_clean, 'WS1')
rpi_agg = aggregate_minute(rpi_clean,'RPI')

# ------------------------------------------------
# 6. MERGE RPi WITH GS1 & WS1 AMBIENT LIGHT
# ------------------------------------------------
fused = (
    rpi_agg
    .merge(gs1_agg, on='minute_ts', how='inner')
    .merge(ws1_agg[['minute_ts','Ambient_Light_WS1']],
           on='minute_ts', how='inner')
)
if fused.empty:
    print("WARNING: No overlapping timestamps. Exiting.")
    exit()
fused.set_index('minute_ts', inplace=True)

# Check for and drop nulls
print(f" fused nulls: {fused.isnull().sum()}")
fused = fused.dropna()

# ------------------------------------------------
# 7. CALIBRATE WITH LINEAR REGRESSION (RPi → GS1)
# ------------------------------------------------
# Temperature
temp_model = LinearRegression()
X_temp = fused[['Temperature_RPI']]
y_temp = fused['Temperature_GS1']
temp_model.fit(X_temp, y_temp)
fused['Temperature_RPI_Calibrated'] = temp_model.predict(X_temp)

# Humidity
hum_model = LinearRegression()
X_hum = fused[['Humidity_RPI']]
y_hum = fused['Humidity_GS1']
hum_model.fit(X_hum, y_hum)
fused['Humidity_RPI_Calibrated'] = hum_model.predict(X_hum)

print(f"Temp calibration: slope={temp_model.coef_[0]:.3f}, intercept={temp_model.intercept_:.3f}")
print(f"Hum calibration:  slope={hum_model.coef_[0]:.3f}, intercept={hum_model.intercept_:.3f}")

# ------------------------------------------------
# 8. VISUALIZATIONS → PDF
# ------------------------------------------------
features = {
    'Temperature':   ['Temperature_RPI','Temperature_GS1','Temperature_RPI_Calibrated'],
    'Humidity':      ['Humidity_RPI','Humidity_GS1','Humidity_RPI_Calibrated'],
    'Ambient_Light': ['Ambient_Light_RPI','Ambient_Light_WS1']
}

with PdfPages(pdf_path) as pdf:
    # 8.1 Histograms
    for feat, cols in features.items():
        for col in cols:
            plt.figure()
            plt.hist(fused[col].dropna(), bins=30)
            plt.title(f"Histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig(); plt.close()

    # 8.2 KDE Plots
    for feat, cols in features.items():
        plt.figure()
        for col in cols:
            sns.kdeplot(fused[col].dropna(), label=col, shade=True)
        plt.title(f"KDE: {feat} by Source")
        plt.xlabel(feat.replace('_',' '))
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(); plt.close()

    # 8.3 Violin Plots
    for feat, cols in features.items():
        melt = fused[cols].melt(var_name='Source', value_name=feat)
        plt.figure(figsize=(8,6))
        sns.violinplot(x='Source', y=feat, data=melt)
        plt.title(f"Violin: {feat} by Source")
        plt.xlabel("Source")
        plt.ylabel(feat.replace('_',' '))
        plt.tight_layout()
        pdf.savefig(); plt.close()

    # 8.4 Correlation Matrix
    corr_mat = fused[np.concatenate(list(features.values()))].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={'label':'Pearson r'})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # 8.5 Scatter Comparisons per Feature
    for feat, cols in features.items():
        plt.figure()
        if feat == 'Ambient_Light':
            sns.scatterplot(x=cols[0], y=cols[1], data=fused, alpha=0.6)
            plt.xlabel("Ambient Light RPi (lux)")
            plt.ylabel("Ambient Light WS1 (lux)")
            plt.title("Ambient Light: RPi vs WS1")
        else:
            xcol,ycol = f"{feat}_RPI", f"{feat}_GS1"
            sns.scatterplot(x=xcol, y=ycol, data=fused, alpha=0.6)
            # overlay calibration line
            model = temp_model if feat=='Temperature' else hum_model
            X = fused[[xcol]]
            plt.plot(X, model.predict(X), color='red')
            plt.xlabel(f"{feat} RPi")
            plt.ylabel(f"{feat} GS1")
            plt.title(f"{feat}: RPi vs GS1 (with calibration)")
        plt.tight_layout()
        pdf.savefig(); plt.close()

    # 8.6 Time Series per Feature
    for feat, cols in features.items():
        plt.figure(figsize=(12,6))
        fused[cols].plot()
        plt.title(f"{feat} Over Time")
        plt.xlabel("Time (minute resolution)")
        plt.ylabel(feat.replace('_',' '))
        plt.legend()
        plt.tight_layout()
        pdf.savefig(); plt.close()

print(f"\n✅ All figures saved to: {pdf_path}")
