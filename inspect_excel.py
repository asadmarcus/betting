import os
import pandas as pd

def inspect_excel(data_folder):
    years = sorted([name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))])
    for year in years:
        file_path = os.path.join(data_folder, year, f'all-euro-data-{year}.xlsx')
        if os.path.exists(file_path):
            print(f"\nReading file: {file_path}")
            excel_data = pd.ExcelFile(file_path)
            sheet_names = excel_data.sheet_names
            print(f"Sheet names: {sheet_names}")
            for sheet in sheet_names:
                print(f"\nSheet: {sheet}")
                df = pd.read_excel(excel_data, sheet_name=sheet)
                print(df.head())
        else:
            print(f"File not found: {file_path}")

data_folder = 'data'
inspect_excel(data_folder)
