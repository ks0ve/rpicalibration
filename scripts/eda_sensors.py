import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesEDA:
    def __init__(self, csv_path, datetime_col=None, date_format=None):
        """
        Initialize EDA class for time series sensor data
        
        Args:
            csv_path (str): Path to CSV file
            datetime_col (str): Name of datetime column (auto-detected if None)
            date_format (str): Date format string (auto-parsed if None)
        """
        self.df = pd.read_csv(csv_path)
        self.datetime_col = datetime_col
        self.date_format = date_format
        self.numeric_cols = []
        self.categorical_cols = []
        
    def preprocess_data(self):
        """Detect and convert datetime column, identify data types"""
        print("=== DATA PREPROCESSING ===")
        
        # Auto-detect datetime column if not specified
        if self.datetime_col is None:
            date_candidates = []
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                    date_candidates.append(col)
            
            if date_candidates:
                self.datetime_col = date_candidates[0]
                print(f"Auto-detected datetime column: {self.datetime_col}")
            else:
                # Try to convert first column or find datetime-like data
                for col in self.df.columns:
                    try:
                        pd.to_datetime(self.df[col].head())
                        self.datetime_col = col
                        print(f"Detected datetime column: {col}")
                        break
                    except:
                        continue
        
        # Convert datetime column
        if self.datetime_col and self.datetime_col in self.df.columns:
            try:
                self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col], format=self.date_format)
                self.df = self.df.sort_values(self.datetime_col)
                self.df.set_index(self.datetime_col, inplace=True)
                print(f"Successfully converted {self.datetime_col} to datetime index")
            except Exception as e:
                print(f"Error converting datetime: {e}")
        
        # Identify numeric and categorical columns
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric columns: {self.numeric_cols}")
        print(f"Categorical columns: {self.categorical_cols}")
        print()
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=== BASIC DATASET INFO ===")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Time span: {self.df.index.max() - self.df.index.min()}")
        print()
        
        print("Data types:")
        print(self.df.dtypes)
        print()
        
        print("Missing values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        print()
    
    def statistical_summary(self):
        """Generate statistical summary for numeric columns"""
        print("=== STATISTICAL SUMMARY ===")
        if self.numeric_cols:
            print(self.df[self.numeric_cols].describe())
            print()
            
            # Additional statistics
            print("Additional Statistics:")
            for col in self.numeric_cols:
                data = self.df[col].dropna()
                print(f"\n{col}:")
                print(f"  Skewness: {data.skew():.3f}")
                print(f"  Kurtosis: {data.kurtosis():.3f}")
                print(f"  Variance: {data.var():.3f}")
                print(f"  Range: {data.max() - data.min():.3f}")
        print()
    
    def detect_anomalies(self):
        """Detect outliers using IQR method"""
        print("=== ANOMALY DETECTION ===")
        outliers_summary = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
        
        for col, stats in outliers_summary.items():
            print(f"{col}: {stats['count']} outliers ({stats['percentage']:.1f}%)")
        print()
    
    def time_series_patterns(self):
        """Analyze time series patterns"""
        print("=== TIME SERIES PATTERNS ===")
        
        # Sampling frequency
        if len(self.df) > 1:
            time_diff = self.df.index.to_series().diff().dropna()
            most_common_freq = time_diff.mode()[0] if not time_diff.mode().empty else None
            print(f"Most common sampling interval: {most_common_freq}")
            
            # Check for gaps
            expected_freq = pd.infer_freq(self.df.index)
            print(f"Inferred frequency: {expected_freq}")
            
            if expected_freq:
                expected_index = pd.date_range(start=self.df.index.min(), 
                                             end=self.df.index.max(), 
                                             freq=expected_freq)
                missing_timestamps = expected_index.difference(self.df.index)
                print(f"Missing timestamps: {len(missing_timestamps)}")
        
        # Trend analysis (simple)
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 1:
                trend = np.polyfit(range(len(data)), data, 1)[0]
                print(f"{col} trend: {'↑' if trend > 0 else '↓'} ({trend:.6f})")
        print()
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("=== GENERATING VISUALIZATIONS ===")
        
        n_numeric = len(self.numeric_cols)
        if n_numeric == 0:
            print("No numeric columns found for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Time series plots
        fig, axes = plt.subplots(n_numeric, 1, figsize=(12, 3*n_numeric))
        if n_numeric == 1:
            axes = [axes]
        
        for i, col in enumerate(self.numeric_cols):
            axes[i].plot(self.df.index, self.df[col], alpha=0.7)
            axes[i].set_title(f'{col} Over Time')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Distribution plots
        fig, axes = plt.subplots(2, (n_numeric + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_numeric > 1 else [axes]
        
        for i, col in enumerate(self.numeric_cols):
            # Histogram
            axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_numeric, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 3. Box plots for outlier detection
        if n_numeric > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            self.df[self.numeric_cols].boxplot(ax=ax)
            ax.set_title('Box Plots - Outlier Detection')
            ax.set_ylabel('Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # 4. Correlation heatmap
        if n_numeric > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            plt.show()
        
        # 5. Rolling statistics (if enough data)
        if len(self.df) > 50:
            fig, axes = plt.subplots(n_numeric, 1, figsize=(12, 3*n_numeric))
            if n_numeric == 1:
                axes = [axes]
            
            window = min(24, len(self.df) // 10)  # Adaptive window size
            
            for i, col in enumerate(self.numeric_cols):
                data = self.df[col].dropna()
                rolling_mean = data.rolling(window=window).mean()
                rolling_std = data.rolling(window=window).std()
                
                axes[i].plot(data.index, data, alpha=0.3, label='Original')
                axes[i].plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean ({window})')
                axes[i].fill_between(rolling_mean.index, 
                                   rolling_mean - rolling_std,
                                   rolling_mean + rolling_std, 
                                   alpha=0.2, label='±1 Std')
                axes[i].set_title(f'{col} - Rolling Statistics')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def run_full_analysis(self):
        """Run complete EDA analysis"""
        print("Starting Time Series Sensor Data EDA...")
        print("=" * 50)
        
        self.preprocess_data()
        self.basic_info()
        self.statistical_summary()
        self.detect_anomalies()
        self.time_series_patterns()
        self.create_visualizations()
        
        print("=" * 50)
        print("EDA Complete!")

# Usage example
if __name__ == "__main__":
    # Initialize and run EDA
    # Replace 'your_file.csv' with your actual file path
    path = "data/outside/"
    eda = TimeSeriesEDA(path+"gs1_out.csv", datetime_col='Timestamp')
    
    # For custom datetime column specification:
    # eda = TimeSeriesEDA('your_file.csv', datetime_col='timestamp', date_format='%Y-%m-%d %H:%M:%S')
    
    eda.run_full_analysis()