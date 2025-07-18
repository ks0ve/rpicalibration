import os
import requests
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

# Modify variables in get_user_inputs() as needed
def get_user_inputs():
    """Change paths and filenames as needed"""
    output_directory = input_directory = 'outside' # Change to your desired output/input directory. Input directory is where the data files are located.
    output_filename = 'train.pdf' # Change to your desired output filename
    gs1_file = 'gs1_out.csv' # Change to your GS1 data file
    ws1_file = 'ws1_out.csv' # Change to your WS1 data file
    rpi_file = 'rpi_out.csv' # Change to your RPi data file
    return {
        'outdir': f"results/{output_directory}",
        'outfile': output_filename,
        'relpath': f"data/{input_directory}",
        'file_gs1': gs1_file,
        'file_ws1': ws1_file,
        'file_raspi': rpi_file
    }

def safe_datetime_parse(filepath, timestamp_col):
    """Safely parse datetime with multiple format attempts"""
    try:
        # First, try reading without parsing
        df = pd.read_csv(filepath)
        
        if timestamp_col not in df.columns:
            print(f"Warning: '{timestamp_col}' not found in {filepath}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Try different datetime parsing approaches
        try:
            # Method 1: pd.to_datetime with UTC inference
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except:
            try:
                # Method 2: pd.to_datetime without UTC
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except:
                print(f"Error: Could not parse timestamps in {filepath}")
                return None
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_and_validate_data(config):
    """Load data with proper error handling and validation"""
    print("\n=== Loading Data ===")
    
    # Load GS1 data
    gs1_path = f"{config['relpath']}/{config['file_gs1']}"
    gs1 = safe_datetime_parse(gs1_path, 'created_at')
    if gs1 is None:
        return None, None, None
    
    # Load WS1 data
    ws1_path = f"{config['relpath']}/{config['file_ws1']}"
    ws1 = safe_datetime_parse(ws1_path, 'created_at')
    if ws1 is None:
        return None, None, None
    
    # Load RPi data
    rpi_path = f"{config['relpath']}/{config['file_raspi']}"
    rpi = safe_datetime_parse(rpi_path, 'Timestamp')
    if rpi is None:
        return None, None, None
    
    print(f"✅ GS1 data loaded: {len(gs1)} rows")
    print(f"✅ WS1 data loaded: {len(ws1)} rows")
    print(f"✅ RPi data loaded: {len(rpi)} rows")
    
    return gs1, ws1, rpi

def clean_sensor_data(gs1, ws1, rpi):
    """Clean and standardize sensor data"""
    print("\n=== Cleaning Data ===")
    
    # GS1: Extract temperature and humidity with flexible column matching
    gs1_temp_cols = [col for col in gs1.columns if 'temp' in col.lower() or 'field1' in col.lower()]
    gs1_hum_cols = [col for col in gs1.columns if 'hum' in col.lower() or 'field2' in col.lower()]
    
    if not gs1_temp_cols or not gs1_hum_cols:
        print("Warning: Could not find temperature/humidity columns in GS1 data")
        print(f"Available columns: {list(gs1.columns)}")
        return None, None, None
    
    gs1_clean = gs1[['created_at', gs1_temp_cols[0], gs1_hum_cols[0]]].copy()
    gs1_clean.rename(columns={
        'created_at': 'Timestamp',
        gs1_temp_cols[0]: 'Temperature (°C)',
        gs1_hum_cols[0]: 'Humidity (%)'
    }, inplace=True)
    
    # WS1: Extract ambient light
    ws1_light_cols = [col for col in ws1.columns if 'light' in col.lower() or 'field3' in col.lower()]
    if not ws1_light_cols:
        print("Warning: Could not find light column in WS1 data")
        print(f"Available columns: {list(ws1.columns)}")
        return None, None, None
    
    ws1_clean = ws1[['created_at', ws1_light_cols[0]]].copy()
    ws1_clean.rename(columns={
        'created_at': 'Timestamp',
        ws1_light_cols[0]: 'Ambient Light (lux)'
    }, inplace=True)
    
    # RPi: Validate required columns exist
    required_rpi_cols = ['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Ambient Light (lux)']
    missing_cols = [col for col in required_rpi_cols if col not in rpi.columns]
    if missing_cols:
        print(f"Warning: Missing columns in RPi data: {missing_cols}")
        print(f"Available columns: {list(rpi.columns)}")
        return None, None, None
    
    rpi_clean = rpi[required_rpi_cols].copy()
    
    # Remove invalid/extreme values
    for df, name in [(gs1_clean, 'GS1'), (rpi_clean, 'RPi')]:
        if 'Temperature (°C)' in df.columns:
            temp_mask = (df['Temperature (°C)'] >= -50) & (df['Temperature (°C)'] <= 100)
            removed_temp = len(df) - temp_mask.sum()
            if removed_temp > 0:
                print(f"Removed {removed_temp} invalid temperature readings from {name}")
            df = df[temp_mask]
        
        if 'Humidity (%)' in df.columns:
            hum_mask = (df['Humidity (%)'] >= 0) & (df['Humidity (%)'] <= 100)
            removed_hum = len(df) - hum_mask.sum()
            if removed_hum > 0:
                print(f"Removed {removed_hum} invalid humidity readings from {name}")
            df = df[hum_mask]
    
    for df, name in [(ws1_clean, 'WS1'), (rpi_clean, 'RPi')]:
        if 'Ambient Light (lux)' in df.columns:
            light_mask = df['Ambient Light (lux)'] >= 0
            removed_light = len(df) - light_mask.sum()
            if removed_light > 0:
                print(f"Removed {removed_light} invalid light readings from {name}")
            df = df[light_mask]
    
    print(f"✅ Cleaned data - GS1: {len(gs1_clean)}, WS1: {len(ws1_clean)}, RPi: {len(rpi_clean)} rows")
    return gs1_clean, ws1_clean, rpi_clean

def normalize_timestamps(gs1_clean, ws1_clean, rpi_clean):
    """Normalize all timestamps to UTC and minute precision"""
    print("\n=== Normalizing Timestamps ===")
    
    # Handle timezone-aware timestamps for GS1 and WS1
    for df, name in [(gs1_clean, 'GS1'), (ws1_clean, 'WS1')]:
        if df['Timestamp'].dt.tz is not None:
            df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        df['minute_ts'] = df['Timestamp'].dt.floor('min')
    
    rpi_clean['Timestamp'] = (
        rpi_clean['Timestamp']
        .dt.tz_localize('America/New_York')   # mark as EDT
        .dt.tz_convert('UTC')                 # to UTC
        .dt.tz_localize(None)                 # drop tz
    )
    
    rpi_clean['minute_ts'] = rpi_clean['Timestamp'].dt.floor('min')
    
    # Print timestamp ranges for debugging
    for df, name in [(gs1_clean, 'GS1'), (ws1_clean, 'WS1'), (rpi_clean, 'RPi')]:
        print(f"{name} timestamp range: {df['minute_ts'].min()} to {df['minute_ts'].max()}")
    
    return gs1_clean, ws1_clean, rpi_clean

def aggregate_by_minute(df, prefix, available_sensors):
    """Aggregate sensor readings by minute with flexible sensor selection"""
    agg_dict = {}
    
    if 'Temperature (°C)' in df.columns and 'temperature' in available_sensors:
        agg_dict['Temperature (°C)'] = 'mean'
    if 'Humidity (%)' in df.columns and 'humidity' in available_sensors:
        agg_dict['Humidity (%)'] = 'mean'
    if 'Ambient Light (lux)' in df.columns and 'light' in available_sensors:
        agg_dict['Ambient Light (lux)'] = 'mean'
    
    if not agg_dict:
        return pd.DataFrame()
    
    grouped = df.groupby('minute_ts').agg(agg_dict).reset_index()
    
    # Rename columns with prefix
    rename_map = {'minute_ts': 'minute_ts'}  # Keep timestamp column name
    for col in agg_dict.keys():
        if col == 'Temperature (°C)':
            rename_map[col] = f'Temperature_{prefix}'
        elif col == 'Humidity (%)':
            rename_map[col] = f'Humidity_{prefix}'
        elif col == 'Ambient Light (lux)':
            rename_map[col] = f'Ambient_Light_{prefix}'
    
    grouped.rename(columns=rename_map, inplace=True)
    return grouped

def merge_sensor_data(gs1_agg, ws1_agg, rpi_agg):
    """Merge sensor data with proper overlap checking"""
    print("\n=== Merging Sensor Data ===")
    
    # Start with RPi as base (assuming it has the most complete data)
    fused = rpi_agg.copy()
    
    # Merge GS1 data
    if not gs1_agg.empty:
        fused = fused.merge(gs1_agg, on='minute_ts', how='inner')
        print(f"After GS1 merge: {len(fused)} overlapping time points")
    
    # Merge WS1 data (only light sensor)
    if not ws1_agg.empty and 'Ambient_Light_WS1' in ws1_agg.columns:
        fused = fused.merge(
            ws1_agg[['minute_ts', 'Ambient_Light_WS1']], 
            on='minute_ts', 
            how='inner'
        )
        print(f"After WS1 merge: {len(fused)} overlapping time points")
    
    if fused.empty:
        print("❌ ERROR: No overlapping timestamps found between sensors!")
        return None
    
    fused.set_index('minute_ts', inplace=True)
    
    # Report data quality
    print(f"Final merged dataset: {len(fused)} time points")
    null_counts = fused.isnull().sum()
    if null_counts.sum() > 0:
        print("Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count}")
    
    # Drop rows with any null values
    fused_clean = fused.dropna()
    if len(fused_clean) < len(fused):
        print(f"Removed {len(fused) - len(fused_clean)} rows with null values")
    
    return fused_clean

def perform_calibration(fused_data):
    """Train calibration models with a train/test split and return performance metrics"""
    # Define models to evaluate
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    }
    calibration_results = {}

    # Identify features based on RPI columns
    features = set(col.replace('_RPI','').lower() for col in fused_data.columns if col.endswith('_RPI'))

    for feat in features:
        col_rpi = f"{feat.title()}_RPI"
        # Determine reference column
        ref_cols = [f"{feat.title()}_GS1", f"{feat.title()}_WS1"]
        col_ref = next((c for c in ref_cols if c in fused_data.columns), None)
        if col_ref is None:
            continue

        # Prepare data and split
        df = fused_data[[col_rpi, col_ref]].dropna()
        X = df[[col_rpi]].values
        y = df[col_ref].values
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )

        calibration_results[feat] = {}
        for method, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store predictions back into the DataFrame for visualization
            pred_col = f"{feat.title()}_RPI_{method}"
            fused_data.loc[idx_test, pred_col] = y_pred

            # Calculate metrics on test set
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            ev = explained_variance_score(y_test, y_pred)

            calibration_results[feat][method] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'explained_variance': ev
            }

    return fused_data, calibration_results

def create_visualizations(fused_data, calibration_results, pdf_path):
    """Create comprehensive visualizations with improved plots"""
    print(f"\n=== Creating Visualizations ===")
    
    # Define available features based on data
    features = {}
    if all(col in fused_data.columns for col in ['Temperature_RPI', 'Temperature_GS1']):
        temp_cols = ['Temperature_RPI', 'Temperature_GS1']
        if 'Temperature_RPI_Calibrated' in fused_data.columns:
            temp_cols.append('Temperature_RPI_Calibrated')
        features['Temperature'] = temp_cols
    
    if all(col in fused_data.columns for col in ['Humidity_RPI', 'Humidity_GS1']):
        hum_cols = ['Humidity_RPI', 'Humidity_GS1']
        if 'Humidity_RPI_Calibrated' in fused_data.columns:
            hum_cols.append('Humidity_RPI_Calibrated')
        features['Humidity'] = hum_cols
    
    if all(col in fused_data.columns for col in ['Ambient_Light_RPI', 'Ambient_Light_WS1']):
        features['Ambient_Light'] = ['Ambient_Light_RPI', 'Ambient_Light_WS1']
    
    with PdfPages(pdf_path) as pdf:
        
         # 5. Time series subplots: one panel per model
        if len(fused_data) > 1:
            for feat in ['Temperature', 'Humidity']:
                methods = list(calibration_results.get(feat.lower(), {}).keys())
                if not methods:
                    continue

                raw_col = f"{feat}_RPI"
                ref_col = f"{feat}_GS1"

                # set up one row per method
                fig, axes = plt.subplots(
                    nrows=len(methods),
                    ncols=1,
                    figsize=(15, 4 * len(methods)),
                    sharex=True
                )

                for ax, method in zip(axes, methods):
                    pred_col = f"{feat}_RPI_{method}"
                    # plot raw & reference
                    ax.plot(fused_data.index, fused_data[raw_col],
                            label='Raw RPI', linewidth=1, alpha=0.7)
                    ax.plot(fused_data.index, fused_data[ref_col],
                            label='Reference GS1', linewidth=1, alpha=0.7)
                    # plot this model’s prediction
                    ax.plot(fused_data.index, fused_data[pred_col],
                            label=method.replace('_', ' ').title(),
                            linewidth=1.5)

                    ax.set_ylabel(feat)
                    ax.set_title(f"{feat}: Raw vs Reference vs {method.title()}")
                    ax.legend(loc='upper left')
                    ax.grid(True, alpha=0.3)

                # common x-label on the bottom subplot
                axes[-1].set_xlabel("Time")
                plt.xticks(rotation=45)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        # —————————————————————————————————————————————
        #  X. Test-set only time series: Reference vs Predicted
        # —————————————————————————————————————————————
        for feat, methods in calibration_results.items():
            # pick the right reference column (GS1 or WS1)
            ref_col = next((c for c in fused_data.columns 
                            if c.startswith(feat.title()) and ('_GS1' in c or '_WS1' in c)),
                           None)
            if ref_col is None:
                continue

            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if pred_col not in fused_data.columns:
                    continue

                # only test‐split rows have a non-null pred_col
                test_df = fused_data[fused_data[pred_col].notnull()]
                if test_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(test_df.index, test_df[ref_col],
                        label='Reference', linewidth=2, alpha=0.8)
                ax.plot(test_df.index, test_df[pred_col],
                        label=method.replace('_', ' ').title(),
                        linewidth=1.5, alpha=0.8)

                ax.set_title(f"{feat.title()} – Test Set: Reference vs {method.replace('_',' ').title()}")
                ax.set_xlabel("Time")
                ax.set_ylabel(feat.title())
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        # 6. Calibration metrics summary table
        fig, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * sum(len(m) for m in calibration_results.values())))
        ax.axis('tight'); ax.axis('off')
        metrics_data = []
        for feat, methods in calibration_results.items():
            for method, res in methods.items():
                metrics_data.append([
                    feat.title(), method.replace('_', ' ').title(),
                    f"{res['r2']:.3f}", f"{res['mse']:.3f}",
                    f"{res['rmse']:.3f}", f"{res['mae']:.3f}", f"{res['explained_variance']:.3f}"
                ])
        cols = ['Feature', 'Model', 'R²', 'MSE', 'RMSE', 'MAE', 'Explained Var']
        table = ax.table(
            cellText=metrics_data, colLabels=cols, cellLoc='center', loc='center'
        )
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
        plt.title('Calibration Model Performance', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # 7. Residual plots for each feature/model
        for feat, methods in calibration_results.items():
            ref_col = f"{feat.title()}_GS1" if feat.title() + '_GS1' in fused_data.columns else None
            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if ref_col and pred_col in fused_data.columns:
                    residuals = fused_data[pred_col] - fused_data[ref_col]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(fused_data[pred_col], residuals, alpha=0.6)
                    ax.axhline(0, linestyle='--', linewidth=1, alpha=0.7)
                    ax.set_xlabel('Predicted Value')
                    ax.set_ylabel('Residual (Pred - Actual)')
                    ax.set_title(f'Residuals: {feat.title()} - {method.replace('_', ' ').title()}')
                    ax.grid(True, alpha=0.3)
                    pdf.savefig(fig); plt.close()
        
        # inside your PdfPages context, after residual plots
        for feat, methods in calibration_results.items():
            ref_col = next((c for c in fused_data.columns if c.startswith(feat.title()) and ('_GS1' in c or '_WS1' in c)), None)
            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if ref_col and pred_col in fused_data.columns:
                    fig, ax = plt.subplots(figsize=(8,6))
                    ax.scatter(fused_data[pred_col], fused_data[ref_col], alpha=0.6)
                    # 45° line
                    minv = min(fused_data[pred_col].min(), fused_data[ref_col].min())
                    maxv = max(fused_data[pred_col].max(), fused_data[ref_col].max())
                    ax.plot([minv, maxv], [minv, maxv], linestyle='--')
                    ax.set_title(f'Predicted vs Actual: {feat.title()} - {method.replace("_"," ").title()}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    pdf.savefig(fig); plt.close()
        
        for feat, methods in calibration_results.items():
            ref_col = next((c for c in fused_data.columns if c.startswith(feat.title()) and ('_GS1' in c or '_WS1' in c)), None)
            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if ref_col and pred_col in fused_data.columns:
                    residuals = fused_data[pred_col] - fused_data[ref_col]
                    fig, ax = plt.subplots(figsize=(8,5))
                    ax.hist(residuals.dropna(), bins=30, alpha=0.7)
                    ax.axvline(0, linestyle='--')
                    ax.set_title(f'Residuals Histogram: {feat.title()} - {method.replace("_"," ").title()}')
                    ax.set_xlabel('Residual')
                    ax.set_ylabel('Frequency')
                    pdf.savefig(fig); plt.close()
        
        # plot a sample day or the entire test window
        times = fused_data.index
        for feat, methods in calibration_results.items():
            ref_col = next((c for c in fused_data.columns if c.startswith(feat.title()) and ('_GS1' in c or '_WS1' in c)), None)
            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if ref_col and pred_col in fused_data.columns:
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(times, fused_data[ref_col], label='Reference')
                    ax.plot(times, fused_data[pred_col], label=method.replace('_',' ').title(), alpha=0.8)
                    ax.set_title(f'Time Series: {feat.title()} - {method.replace("_"," ").title()}')
                    ax.set_xlabel('Timestamp')
                    ax.set_ylabel(feat.title())
                    ax.legend()
                    pdf.savefig(fig); plt.close()
        
        fused_data['hour'] = fused_data.index.hour
        for feat, methods in calibration_results.items():
            ref_col = next((c for c in fused_data.columns if c.startswith(feat.title()) and ('_GS1' in c or '_WS1' in c)), None)
            for method in methods:
                pred_col = f"{feat.title()}_RPI_{method}"
                if ref_col and pred_col in fused_data.columns:
                    fig, ax = plt.subplots(figsize=(12,5))
                    fused_data['residual'] = fused_data[pred_col] - fused_data[ref_col]
                    fused_data.boxplot(column='residual', by='hour', ax=ax)
                    ax.set_title(f'Hourly Residuals: {feat.title()} - {method.replace("_"," ").title()}')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Residual')
                    plt.suptitle('')
                    pdf.savefig(fig); plt.close()
        

def main():
    """Main execution function"""
    print("🌡️ Environmental Sensor Calibration Tool")
    print("=" * 50)
    
    try:
        # Get user configuration
        config = get_user_inputs()
        
        # Create output directory
        output_dir = f"{config['outdir']}"
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, config['outfile'])
        print(f"Output will be saved to: {pdf_path}")
        
        # Load data
        gs1, ws1, rpi = load_and_validate_data(config)
        if any(df is None for df in [gs1, ws1, rpi]):
            print("❌ Failed to load data. Exiting.")
            return
        
        # Clean data
        gs1_clean, ws1_clean, rpi_clean = clean_sensor_data(gs1, ws1, rpi)
        if any(df is None for df in [gs1_clean, ws1_clean, rpi_clean]):
            print("❌ Failed to clean data. Exiting.")
            return
        
        # Normalize timestamps
        gs1_clean, ws1_clean, rpi_clean = normalize_timestamps(gs1_clean, ws1_clean, rpi_clean)
        
        # Save clean csvs to output directory
        gs1_clean.to_csv(os.path.join(output_dir, 'gs1_clean.csv'), index=False)
        ws1_clean.to_csv(os.path.join(output_dir, 'ws1_clean.csv'), index=False)
        rpi_clean.to_csv(os.path.join(output_dir, 'rpi_clean.csv'), index=False)
        print("✅ Cleaned data saved to output directory")
        # Aggregate by minute
        gs1_agg = aggregate_by_minute(gs1_clean, 'GS1', ['temperature', 'humidity'])
        ws1_agg = aggregate_by_minute(ws1_clean, 'WS1', ['light'])
        rpi_agg = aggregate_by_minute(rpi_clean, 'RPI', ['temperature', 'humidity', 'light'])
        
        # Merge data
        fused_data = merge_sensor_data(gs1_agg, ws1_agg, rpi_agg)
        if fused_data is None:
            print("❌ Failed to merge sensor data. Check timestamp alignment.")
            return
        else:
            # Save merged data to CSV
            fused_data.to_csv(os.path.join(output_dir, 'fused_data.csv'))
            print(f"✅ Merged data saved to: {os.path.join(output_dir, 'fused_data.csv')}")
        # Perform calibration
        fused_data, calibration_results = perform_calibration(fused_data)
        
        # Create visualizations
        create_visualizations(fused_data, calibration_results, pdf_path)
        
        print(f"\n✅ Analysis complete! Results saved to: {pdf_path}")
        print(f"📊 Processed {len(fused_data)} synchronized data points")
        
        # Save calibration equations to file
        cal_file = os.path.join(output_dir, "error_metrics.txt")
        with open(cal_file, 'w') as f:
            f.write("Environmental Sensor Calibration Results\n")
            f.write("=" * 45 + "\n\n")
            
            for param, methods in calibration_results.items():
                f.write(f"{param.title()} Calibration Results\n")
                f.write("-" * 30 + "\n")
                for name, res in methods.items():
                    f.write(f"{name.title()}:\n")
                    f.write(f"  R^2: {res['r2']:.4f}, MSE: {res['mse']:.4f}, "
                            f"RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}, "
                            f"EV: {res['explained_variance']:.4f}\n\n")
            
            f.write("Usage Instructions:\n")
            f.write("To calibrate new RPi readings, apply the equations above:\n")
            f.write("calibrated_value = slope * raw_rpi_value + intercept\n")
        
        print(f"📝 Calibration equations saved to: {cal_file}")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()