import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
file_path = os.path.join(data_dir, "2018Use_data.xlsx")

df = pd.read_excel(file_path, header=1)
print(df.columns.tolist())
print(df.head())
