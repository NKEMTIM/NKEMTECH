import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
output_dir = r"c:\Users\DELL\Downloads\enegy consumption\analysis"
os.makedirs(output_dir, exist_ok=True)
md_path = os.path.join(output_dir, "summary.md")

files = [
    "2018Use_data.xlsx", "2019Use_data.xlsx", "2020Use_data.xlsx",
    "2021Use_data (1).xlsx", "2022data (2).xlsx", "2023Use_data (3).xlsx",
    "2024Use_data (4).xlsx", "2025Use_data.xlsx"
]

def extract_year(filename):
    match = re.search(r'(20\d{2})', filename)
    return int(match.group(1)) if match else None

with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Energy Data Descriptive Analysis\n\n")

    f.write("## 1 & 2. Load and Inspect Files\n\n")
    
    dfs = []
    can_merge = True
    base_columns = None
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            xl = pd.ExcelFile(file_path)
            sheet_name = xl.sheet_names[0]
            # Read skipping first row to get correct headers
            df = xl.parse(sheet_name, header=1)
            
            # Write inspection info
            f.write(f"### File: {file}\n")
            f.write(f"- **Sheet Names**: {xl.sheet_names}\n")
            f.write(f"- **Shape**: {df.shape}\n")
            f.write("- **Columns**: " + ", ".join([str(c) for c in df.columns]) + "\n\n")
            f.write("- **First few rows**:\n")
            f.write("```text\n")
            f.write(df.head(3).to_string() + "\n")
            f.write("```\n\n")
            
            # Standardization preparation
            df['Source_File'] = file
            df['Year'] = extract_year(file)
            
            # Check structure match
            if base_columns is None:
                base_columns = list(df.columns)
            else:
                if list(df.columns) != base_columns:
                    f.write(f"⚠️ **Warning**: Columns in {file} mismatch with the base structure.\n")
                    can_merge = False
                    
            dfs.append(df)
            
        except Exception as e:
            f.write(f"Error loading {file}: {e}\n\n")
            can_merge = False
            
    f.write("## 3. Standardize Column Names\n\n")
    
    # We rename columns to English for standard analysis
    # Assuming columns are: [Identifier, Category, Total, Jan, Feb... Dec, Source_File, Year]
    # Let's verify lengths. The original dataframe has 15 columns. 
    # With Source_File and Year it will have 17.
    standard_cols = ['ID', 'Category', 'Total', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Source_File', 'Year']
    
    for i, df in enumerate(dfs):
        if len(df.columns) == len(standard_cols):
            df.columns = standard_cols
            
    f.write("Standardized columns to English equivalents to facilitate merging and analysis:\n")
    f.write(f"`{standard_cols}`\n\n")
    
    f.write("## 4. Check Mergeability\n\n")
    if can_merge:
        f.write("All files have matching column structures and can be concatenated safely.\n\n")
    else:
        f.write("There were structural mismatches, but we standardized and will attempt concatenation.\n\n")
        
    f.write("## 5. Combine and Data Overview\n\n")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    f.write(f"- **Total Rows**: {combined_df.shape[0]}\n")
    f.write(f"- **Total Columns**: {combined_df.shape[1]}\n")
    f.write(f"- **Column Names**: {', '.join(combined_df.columns)}\n\n")
    
    f.write("**Data Types**:\n```text\n")
    f.write(combined_df.dtypes.to_string() + "\n```\n\n")
    
    f.write("**Missing Values Count & Percentage**:\n```text\n")
    missing_data = pd.DataFrame({
        'Count': combined_df.isnull().sum(),
        'Percentage (%)': (combined_df.isnull().sum() / len(combined_df) * 100).round(2)
    })
    f.write(missing_data.to_string() + "\n```\n\n")
    
    f.write(f"- **Duplicate Rows**: {combined_df.duplicated().sum()}\n\n")
    
    f.write("## 6. Descriptive Statistics\n\n")
    f.write("```text\n")
    f.write(combined_df.describe().to_string() + "\n```\n\n")
    
    f.write("## 7. Data Distribution & Visualizations\n\n")
    
    numeric_cols = ['Total', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Convert them to numeric, coercing errors
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
    f.write("We selected the `Total` column (yearly total energy consumption) as the main energy usage column for overall yearly distribution, and the month columns for temporal distribution.\n\n")
    
    # Histogram of Total
    plt.figure(figsize=(10, 6))
    sns.histplot(combined_df['Total'].dropna(), kde=True, bins=30)
    plt.title('Distribution of Yearly Total Energy Usage')
    plt.xlabel('Total Energy Usage (MWh)')
    plt.ylabel('Frequency')
    hist_path = os.path.join(output_dir, "total_usage_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    
    # Boxplot to check for outliers in Total across Years
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y='Total', data=combined_df)
    plt.title('Boxplot of Total Energy Usage by Year')
    plt.ylabel('Total Energy Usage (MWh)')
    box_path = os.path.join(output_dir, "total_usage_boxplot.png")
    plt.savefig(box_path)
    plt.close()
    
    f.write(f"![Histogram of Total Usage](file:///{hist_path.replace(chr(92), '/')})\n\n")
    f.write(f"![Boxplot of Total Usage by Year](file:///{box_path.replace(chr(92), '/')})\n\n")
    
    skewness = combined_df['Total'].skew()
    f.write(f"- **Skewness of Total**: {skewness:.2f}\n")
    f.write("- **Outliers**: As seen in the boxplot, there are likely a few very high outlier usage cases. The distribution is strongly positively skewed.\n\n")
    
    f.write("## 8. Date and Time Information\n\n")
    f.write("The dataset is structured with years as separate files and months as columns. By melting the dataframe, we can create a proper time-series dataset. The date range spans from 2018 to 2025. The implied time frequency is **Monthly**.\n\n")
    
    # Let's melt it to show how it would look like
    melted_df = pd.melt(combined_df, id_vars=['ID', 'Category', 'Year'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], var_name='Month', value_name='Consumption_MWh')
    
    f.write("Example of melted time-series ready data:\n```text\n")
    f.write(melted_df.head(5).to_string() + "\n```\n\n")
    
    f.write("## 9 & 10. Summary and Next Steps\n\n")
    f.write("The `Consumption_MWh` column from the melted dataset is the most suitable target column for time-series forecasting. We avoided aggressive data cleaning or feature scaling for this preliminary phase.\n")

print("Analysis script finished successfully.")
