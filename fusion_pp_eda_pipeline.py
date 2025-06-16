import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------------
# 1. CREATE OUTPUT DIRECTORY FOR THE PDF
# ------------------------------------------------
outdir = input("Enter output directory within figures folder for plots: ")
outfile = input("Enter output pdf name: ")

output_dir = f"figures/{outdir}"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, f"{outfile}")
print(f"Saving figures to: {output_dir} as '{outfile}'")

relpath = input("Enter data folder name: ")
file_gs1 = input("Enter gs1 data filename: ")
file_ws1 = input("Enter ws1 data filename: ")
file_raspi = input("Enter raspi data filename: ")

# ------------------------------------------------
# 2. LOAD RAW CSVs (WITH ROBUST TIMESTAMP PARSING)
# ------------------------------------------------
gs1 = pd.read_csv(
    f'{relpath}/{file_gs1}',
    parse_dates=['created_at'],
    date_parser=lambda x: pd.to_datetime(x, utc=True)
)
ws1 = pd.read_csv(
    f'{relpath}/{file_ws1}',
    parse_dates=['created_at'],
    date_parser=lambda x: pd.to_datetime(x, utc=True)
)
rpi = pd.read_csv(
    f'{relpath}/{file_raspi}',
    parse_dates=['Timestamp']
)

# ------------------------------------------------
# 3. CLEAN GS1 (KEEP ONLY TIMESTAMP, TEMPERATURE, HUMIDITY)
# ------------------------------------------------
gs1_clean = gs1[['created_at', 'field1(Temperature ºC )', 'field2(Humidity)']].copy()
gs1_clean.rename(
    columns={
        'created_at': 'Timestamp',
        'field1(Temperature ºC )': 'Temperature (°C)',
        'field2(Humidity)': 'Humidity (%)'
    },
    inplace=True
)

# ------------------------------------------------
# 4. CLEAN WS1 (KEEP ONLY TIMESTAMP, TEMPERATURE, HUMIDITY, AMBIENT LIGHT)
# ------------------------------------------------
ws1_clean = ws1[['created_at', 'field1(Temperature ºC )', 
                 'field2(Humidity)', 'field3(Light)']].copy()
ws1_clean.rename(
    columns={
        'created_at': 'Timestamp',
        'field1(Temperature ºC )': 'Temperature (°C)',
        'field2(Humidity)': 'Humidity (%)',
        'field3(Light)': 'Ambient Light (lux)'
    },
    inplace=True
)

# ------------------------------------------------
# 5. CLEAN RPi (KEEP ONLY TIMESTAMP, TEMPERATURE, HUMIDITY, AMBIENT LIGHT)
# ------------------------------------------------
rpi_clean = rpi[['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Ambient Light (lux)']].copy()

# ------------------------------------------------
# 6. NORMALIZE TIMESTAMPS TO MINUTE PRECISION
# ------------------------------------------------
for df in (gs1_clean, ws1_clean):
    if df['Timestamp'].dt.tz is not None:
        df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    df['minute_ts'] = df['Timestamp'].dt.floor('min')

rpi_clean['Timestamp'] = rpi_clean['Timestamp'].dt.tz_localize(None)
rpi_clean['minute_ts'] = rpi_clean['Timestamp'].dt.floor('min')

# ------------------------------------------------
# 7. AGGREGATE READINGS PER MINUTE (MEAN IF MULTIPLE)
# ------------------------------------------------
def aggregate_minute(df, prefix, include_light=False):
    agg_dict = {
        'Temperature (°C)': 'mean',
        'Humidity (%)': 'mean'
    }
    if include_light:
        agg_dict['Ambient Light (lux)'] = 'mean'
    
    grouped = df.groupby('minute_ts').agg(agg_dict).reset_index()
    rename_map = {
        'Temperature (°C)': f'Temperature_{prefix}',
        'Humidity (%)': f'Humidity_{prefix}'
    }
    if include_light:
        rename_map['Ambient Light (lux)'] = f'Ambient_Light_{prefix}'
    
    grouped.rename(columns=rename_map, inplace=True)
    return grouped

gs1_agg = aggregate_minute(gs1_clean, 'GS1', include_light=False)
ws1_agg = aggregate_minute(ws1_clean, 'WS1', include_light=True)
rpi_agg = aggregate_minute(rpi_clean, 'RPI', include_light=True)

# ------------------------------------------------
# 8. COMPUTE WS1–GS1 AVERAGED METRICS (TEMP & HUMIDITY)
# ------------------------------------------------
ws1_gs1_merge = ws1_agg.merge(gs1_agg, on='minute_ts', how='inner')
ws1_gs1_merge['Temperature_WS1_GS1_Avg'] = (
    ws1_gs1_merge['Temperature_WS1'] + ws1_gs1_merge['Temperature_GS1']
) / 2
ws1_gs1_merge['Humidity_WS1_GS1_Avg'] = (
    ws1_gs1_merge['Humidity_WS1'] + ws1_gs1_merge['Humidity_GS1']
) / 2

# ------------------------------------------------
# 9. MERGE RPi WITH WS1, GS1, AND AVERAGED METRICS
# ------------------------------------------------
merge_individual = (
    rpi_agg
    .merge(ws1_agg, on='minute_ts', how='inner')
    .merge(gs1_agg, on='minute_ts', how='inner')
)
fused = merge_individual.merge(
    ws1_gs1_merge[['minute_ts', 'Temperature_WS1_GS1_Avg', 'Humidity_WS1_GS1_Avg']],
    on='minute_ts',
    how='inner'
)

if fused.empty:
    print("\nWARNING: No overlapping minutes found. Aborting PDF creation.")
else:
    fused.set_index('minute_ts', inplace=True)

    # ------------------------------------------------
    # 10. SAVE ALL PLOTS (INCL. VIOLIN + KDE) TO PDF
    # ------------------------------------------------
    with PdfPages(pdf_path) as pdf:
        # 10.1 Histograms
        for col in fused.columns:
            plt.figure()
            plt.hist(fused[col].dropna(), bins=30)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 10.2 Violin: Temperature
        temp_df = fused[['Temperature_RPI', 'Temperature_WS1', 
                         'Temperature_GS1', 'Temperature_WS1_GS1_Avg']].copy()
        temp_long = temp_df.melt(var_name='Device', value_name='Temperature')
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Device', y='Temperature', data=temp_long)
        plt.title("Temperature Distribution by Device")
        plt.xlabel("Device")
        plt.ylabel("Temperature (°C)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.3 Violin: Humidity
        hum_df = fused[['Humidity_RPI', 'Humidity_WS1', 
                        'Humidity_GS1', 'Humidity_WS1_GS1_Avg']].copy()
        hum_long = hum_df.melt(var_name='Device', value_name='Humidity')
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Device', y='Humidity', data=hum_long)
        plt.title("Humidity Distribution by Device")
        plt.xlabel("Device")
        plt.ylabel("Humidity (%)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.4 Violin: Ambient Light
        light_df = fused[['Ambient_Light_RPI', 'Ambient_Light_WS1']].copy()
        light_long = light_df.melt(var_name='Device', value_name='Ambient Light (lux)')
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Device', y='Ambient Light (lux)', data=light_long)
        plt.title("Ambient Light Distribution by Device")
        plt.xlabel("Device")
        plt.ylabel("Ambient Light (lux)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.5 KDE Overlays: Temperature
        plt.figure(figsize=(8, 6))
        sns.kdeplot(fused['Temperature_RPI'], label='RPi', shade=True)
        sns.kdeplot(fused['Temperature_WS1'], label='WS1', shade=True)
        sns.kdeplot(fused['Temperature_GS1'], label='GS1', shade=True)
        sns.kdeplot(fused['Temperature_WS1_GS1_Avg'], label='WS1-GS1 Avg', shade=True)
        plt.title("KDE: Temperature by Device")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.6 KDE Overlays: Humidity
        plt.figure(figsize=(8, 6))
        sns.kdeplot(fused['Humidity_RPI'], label='RPi', shade=True)
        sns.kdeplot(fused['Humidity_WS1'], label='WS1', shade=True)
        sns.kdeplot(fused['Humidity_GS1'], label='GS1', shade=True)
        sns.kdeplot(fused['Humidity_WS1_GS1_Avg'], label='WS1-GS1 Avg', shade=True)
        plt.title("KDE: Humidity by Device")
        plt.xlabel("Humidity (%)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.7 KDE Overlays: Ambient Light
        plt.figure(figsize=(8, 6))
        sns.kdeplot(fused['Ambient_Light_RPI'], label='RPi', shade=True)
        sns.kdeplot(fused['Ambient_Light_WS1'], label='WS1', shade=True)
        plt.title("KDE: Ambient Light by Device")
        plt.xlabel("Ambient Light (lux)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.8 Correlation Heatmap
        corr_matrix = fused.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'label': 'Pearson r'}
        )
        plt.title("Correlation Matrix (Fused Data with Averages)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.9 Scatter: Temperature Comparisons
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x='Temperature_RPI', y='Temperature_WS1',
            data=fused, label='WS1', alpha=0.6
        )
        sns.scatterplot(
            x='Temperature_RPI', y='Temperature_GS1',
            data=fused, label='GS1', alpha=0.6
        )
        sns.scatterplot(
            x='Temperature_RPI', y='Temperature_WS1_GS1_Avg',
            data=fused, label='WS1-GS1 Avg', alpha=0.6
        )
        plt.legend()
        plt.title("RPi Temperature vs Ubibot (WS1, GS1, WS1-GS1 Avg)")
        plt.xlabel("RPi Temperature (°C)")
        plt.ylabel("Ubibot Temperature (°C)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.10 Scatter: Humidity Comparisons
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x='Humidity_RPI', y='Humidity_WS1',
            data=fused, label='WS1', alpha=0.6
        )
        sns.scatterplot(
            x='Humidity_RPI', y='Humidity_GS1',
            data=fused, label='GS1', alpha=0.6
        )
        sns.scatterplot(
            x='Humidity_RPI', y='Humidity_WS1_GS1_Avg',
            data=fused, label='WS1-GS1 Avg', alpha=0.6
        )
        plt.legend()
        plt.title("RPi Humidity vs Ubibot (WS1, GS1, WS1-GS1 Avg)")
        plt.xlabel("RPi Humidity (%)")
        plt.ylabel("Ubibot Humidity (%)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.11 Scatter: Ambient Light Comparison (RPi vs WS1)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            x='Ambient_Light_RPI', y='Ambient_Light_WS1',
            data=fused, alpha=0.6
        )
        plt.title("Ambient Light: RPi vs WS1")
        plt.xlabel("RPi Ambient Light (lux)")
        plt.ylabel("WS1 Ambient Light (lux)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.12 Time-Series: Temperature Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        fused[[
            'Temperature_RPI',
            'Temperature_WS1',
            'Temperature_GS1',
            'Temperature_WS1_GS1_Avg'
        ]].plot(ax=ax)
        ax.set_title("Temperature Over Time (RPi vs WS1 vs GS1 vs Avg)")
        ax.set_xlabel("Time (Minute Resolution)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.13 Time-Series: Humidity Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        fused[[
            'Humidity_RPI',
            'Humidity_WS1',
            'Humidity_GS1',
            'Humidity_WS1_GS1_Avg'
        ]].plot(ax=ax)
        ax.set_title("Humidity Over Time (RPi vs WS1 vs GS1 vs Avg)")
        ax.set_xlabel("Time (Minute Resolution)")
        ax.set_ylabel("Humidity (%)")
        ax.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 10.14 Time-Series: Ambient Light Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        fused[['Ambient_Light_RPI', 'Ambient_Light_WS1']].plot(ax=ax)
        ax.set_title("Ambient Light Over Time (RPi vs WS1)")
        ax.set_xlabel("Time (Minute Resolution)")
        ax.set_ylabel("Ambient Light (lux)")
        ax.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"\n✅ All figures saved to: {pdf_path}")
