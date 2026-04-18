import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
output_file = r"c:\Users\DELL\Downloads\enegy consumption\analysis\output_report.txt"

files = [
    "2018Use_data.xlsx", "2019Use_data.xlsx", "2020Use_data.xlsx",
    "2021Use_data (1).xlsx", "2022data (2).xlsx", "2023Use_data (3).xlsx",
    "2024Use_data (4).xlsx", "2025Use_data.xlsx"
]

with open(output_file, 'w', encoding='utf-8') as f:
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            xl = pd.ExcelFile(file_path)
            sheet_name = xl.sheet_names[0]
            df = xl.parse(sheet_name)
            f.write(f"\nFile: {file}\n")
            f.write(str(df.head(5)) + "\n")
            f.write("-" * 50 + "\n")
        except Exception as e:
            f.write(f"Error loading {file}: {e}\n")
