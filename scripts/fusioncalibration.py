import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

def get_user_inputs():
    """Get all user inputs with validation"""
    print("=== Environmental Sensor Calibration Setup ===")
    
    # Output configuration
    outdir = input("Enter output directory within figures folder for plots: ").strip()
    if not outdir:
        outdir = "sensor_analysis"
        print(f"Using default directory: {outdir}")
    
    outfile = input("Enter output pdf name (e.g., analysis_plots.pdf): ").strip()
    if not outfile.endswith('.pdf'):
        outfile += '.pdf'
    
    # Data configuration
    relpath = input("Enter data folder name: ").strip()
    if not relpath:
        relpath = "data"
        print(f"Using default data folder: {relpath}")
    
    file_gs1 = input("Enter GS1 data filename: ").strip()
    file_ws1 = input("Enter WS1 data filename: ").strip()
    file_raspi = input("Enter RPi data filename: ").strip()
    
    return {
        'outdir': outdir,
        'outfile': outfile,
        'relpath': relpath,
        'file_gs1': file_gs1,
        'file_ws1': file_ws1,
        'file_raspi': file_raspi
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
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
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
    
    # Handle RPi timestamps (may or may not be timezone-aware)
    if rpi_clean['Timestamp'].dt.tz is not None:
        rpi_clean['Timestamp'] = rpi_clean['Timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # Assume RPi timestamps are in local time, convert to UTC if needed
        # You may want to adjust this based on your setup
        pass
    
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
    """Perform linear regression calibration with comprehensive statistics"""
    print("\n=== Performing Calibration ===")
    
    calibration_results = {}
    
    # Temperature calibration
    if all(col in fused_data.columns for col in ['Temperature_RPI', 'Temperature_GS1']):
        temp_model = LinearRegression()
        X_temp = fused_data[['Temperature_RPI']]
        y_temp = fused_data['Temperature_GS1']
        
        temp_model.fit(X_temp, y_temp)
        temp_pred = temp_model.predict(X_temp)
        fused_data['Temperature_RPI_Calibrated'] = temp_pred
        
        # Calculate statistics
        temp_r2 = r2_score(y_temp, temp_pred)
        temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_pred))
        temp_mae = mean_absolute_error(y_temp, temp_pred)
        
        calibration_results['temperature'] = {
            'model': temp_model,
            'slope': temp_model.coef_[0],
            'intercept': temp_model.intercept_,
            'r2': temp_r2,
            'rmse': temp_rmse,
            'mae': temp_mae
        }
        
        print(f"Temperature calibration:")
        print(f"  Equation: y = {temp_model.coef_[0]:.3f}x + {temp_model.intercept_:.3f}")
        print(f"  R² = {temp_r2:.3f}, RMSE = {temp_rmse:.3f}°C, MAE = {temp_mae:.3f}°C")
    
    # Humidity calibration
    if all(col in fused_data.columns for col in ['Humidity_RPI', 'Humidity_GS1']):
        hum_model = LinearRegression()
        X_hum = fused_data[['Humidity_RPI']]
        y_hum = fused_data['Humidity_GS1']
        
        hum_model.fit(X_hum, y_hum)
        hum_pred = hum_model.predict(X_hum)
        fused_data['Humidity_RPI_Calibrated'] = hum_pred
        
        # Calculate statistics
        hum_r2 = r2_score(y_hum, hum_pred)
        hum_rmse = np.sqrt(mean_squared_error(y_hum, hum_pred))
        hum_mae = mean_absolute_error(y_hum, hum_pred)
        
        calibration_results['humidity'] = {
            'model': hum_model,
            'slope': hum_model.coef_[0],
            'intercept': hum_model.intercept_,
            'r2': hum_r2,
            'rmse': hum_rmse,
            'mae': hum_mae
        }
        
        print(f"Humidity calibration:")
        print(f"  Equation: y = {hum_model.coef_[0]:.3f}x + {hum_model.intercept_:.3f}")
        print(f"  R² = {hum_r2:.3f}, RMSE = {hum_rmse:.3f}%, MAE = {hum_mae:.3f}%")
    
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
        # 1. Summary statistics table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = []
        for feat, cols in features.items():
            for col in cols:
                data = fused_data[col].dropna()
                summary_data.append([
                    col, len(data), f"{data.mean():.2f}", f"{data.std():.2f}",
                    f"{data.min():.2f}", f"{data.max():.2f}"
                ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Sensor', 'Count', 'Mean', 'Std', 'Min', 'Max'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('Summary Statistics', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 2. Improved histograms with overlays
        for feat, cols in features.items():
            fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
            if len(cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(cols):
                data = fused_data[col].dropna()
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].axvline(data.mean(), color='red', linestyle='--', 
                               label=f'Mean: {data.mean():.2f}')
                axes[i].set_title(f"{col}")
                axes[i].set_xlabel(col.replace('_', ' '))
                axes[i].set_ylabel("Frequency")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f"Distribution Comparison: {feat}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # 3. Enhanced scatter plots with regression statistics
        for feat, cols in features.items():
            if feat == 'Ambient_Light' and len(cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                x_data = fused_data[cols[0]].dropna()
                y_data = fused_data[cols[1]].dropna()
                
                # Align data for correlation
                common_idx = x_data.index.intersection(y_data.index)
                x_aligned = x_data.loc[common_idx]
                y_aligned = y_data.loc[common_idx]
                
                ax.scatter(x_aligned, y_aligned, alpha=0.6)
                
                # Add correlation coefficient
                if len(x_aligned) > 1:
                    corr = np.corrcoef(x_aligned, y_aligned)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle="round", facecolor='wheat'))
                
                ax.set_xlabel(f"{cols[0].replace('_', ' ')}")
                ax.set_ylabel(f"{cols[1].replace('_', ' ')}")
                ax.set_title(f"Light Sensor Comparison")
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                
            elif feat in ['Temperature', 'Humidity'] and len(cols) >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Raw comparison
                rpi_col = f"{feat}_RPI"
                ref_col = f"{feat}_GS1"
                
                x_data = fused_data[rpi_col]
                y_data = fused_data[ref_col]
                
                axes[0].scatter(x_data, y_data, alpha=0.6, label='Data points')
                
                # Add calibration line if available
                if feat.lower() in calibration_results:
                    cal_result = calibration_results[feat.lower()]
                    x_range = np.linspace(x_data.min(), x_data.max(), 100)
                    y_pred = cal_result['slope'] * x_range + cal_result['intercept']
                    axes[0].plot(x_range, y_pred, 'r-', 
                                label=f"Calibration line (R² = {cal_result['r2']:.3f})")
                
                # Add 1:1 line
                min_val = min(x_data.min(), y_data.min())
                max_val = max(x_data.max(), y_data.max())
                axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', 
                            alpha=0.5, label='1:1 line')
                
                axes[0].set_xlabel(f"{feat} RPi")
                axes[0].set_ylabel(f"{feat} Reference (GS1)")
                axes[0].set_title(f"Raw {feat} Comparison")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Calibrated comparison (if available)
                cal_col = f"{feat}_RPI_Calibrated"
                if cal_col in fused_data.columns:
                    cal_data = fused_data[cal_col]
                    axes[1].scatter(cal_data, y_data, alpha=0.6, color='green')
                    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', 
                                alpha=0.5, label='1:1 line')
                    
                    # Calculate residuals
                    residuals = y_data - cal_data
                    rmse = np.sqrt(np.mean(residuals**2))
                    axes[1].text(0.05, 0.95, f'RMSE = {rmse:.3f}', 
                                transform=axes[1].transAxes,
                                bbox=dict(boxstyle="round", facecolor='lightgreen'))
                    
                    axes[1].set_xlabel(f"{feat} RPi (Calibrated)")
                    axes[1].set_ylabel(f"{feat} Reference (GS1)")
                    axes[1].set_title(f"Calibrated {feat} Comparison")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'Calibration not available', 
                                ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f"Calibrated {feat} (Not Available)")
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        # 4. Time series plots
        if len(fused_data) > 1:
            for feat, cols in features.items():
                fig, ax = plt.subplots(figsize=(15, 6))
                
                for col in cols:
                    data = fused_data[col].dropna()
                    ax.plot(data.index, data.values, label=col.replace('_', ' '), 
                           linewidth=1, alpha=0.8)
                
                ax.set_xlabel("Time")
                ax.set_ylabel(feat.replace('_', ' '))
                ax.set_title(f"{feat} Time Series Comparison")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        # 5. Enhanced correlation matrix
        if len(features) > 0:
            all_cols = []
            for cols in features.values():
                all_cols.extend(cols)
            
            corr_data = fused_data[all_cols]
            corr_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', 
                          vmin=-1, vmax=1)
            
            # Add text annotations
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix)))
            ax.set_xticklabels([col.replace('_', '\n') for col in corr_matrix.columns], 
                              rotation=45, ha='right')
            ax.set_yticklabels([col.replace('_', '\n') for col in corr_matrix.index])
            
            plt.colorbar(im, ax=ax, label='Pearson Correlation Coefficient')
            plt.title("Sensor Correlation Matrix")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

def main():
    """Main execution function"""
    print("🌡️ Environmental Sensor Calibration Tool")
    print("=" * 50)
    
    try:
        # Get user configuration
        config = get_user_inputs()
        
        # Create output directory
        output_dir = f"figures/{config['outdir']}"
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
        
        # Aggregate by minute
        gs1_agg = aggregate_by_minute(gs1_clean, 'GS1', ['temperature', 'humidity'])
        ws1_agg = aggregate_by_minute(ws1_clean, 'WS1', ['light'])
        rpi_agg = aggregate_by_minute(rpi_clean, 'RPI', ['temperature', 'humidity', 'light'])
        
        # Merge data
        fused_data = merge_sensor_data(gs1_agg, ws1_agg, rpi_agg)
        if fused_data is None:
            print("❌ Failed to merge sensor data. Check timestamp alignment.")
            return
        
        # Perform calibration
        fused_data, calibration_results = perform_calibration(fused_data)
        
        # Create visualizations
        create_visualizations(fused_data, calibration_results, pdf_path)
        
        print(f"\n✅ Analysis complete! Results saved to: {pdf_path}")
        print(f"📊 Processed {len(fused_data)} synchronized data points")
        
        # Save calibration equations to file
        cal_file = os.path.join(output_dir, "calibration_equations.txt")
        with open(cal_file, 'w') as f:
            f.write("Environmental Sensor Calibration Results\n")
            f.write("=" * 45 + "\n\n")
            
            for param, results in calibration_results.items():
                f.write(f"{param.title()} Calibration:\n")
                f.write(f"  Equation: y = {results['slope']:.6f} * x + {results['intercept']:.6f}\n")
                f.write(f"  R-squared: {results['r2']:.4f}\n")
                f.write(f"  RMSE: {results['rmse']:.4f}\n")
                f.write(f"  MAE: {results['mae']:.4f}\n\n")
            
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