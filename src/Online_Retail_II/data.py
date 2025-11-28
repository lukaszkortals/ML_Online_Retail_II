from openpyxl import load_workbook
import pandas as pd
import kagglehub as kh
import os

def load_df_from_kaggle(max_rows = 1000) -> pd.DataFrame:

    file_path = kh.dataset_download("shandeep777/online-retail-ii")

    xlsx_files = [f for f in os.listdir(file_path) if f.lower().endswith(".xlsx")]

    if len(xlsx_files) == 0:
        raise RuntimeError("Brak plików")

    xlsx_path = os.path.join(file_path, xlsx_files[0])

    #workbook pod streeming zeby nie wieszalo
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)

    sheet1 = wb["Year 2009-2010"]
    sheet2 = wb["Year 2010-2011"]

    #ładowanie pliku strimingiem bo inaczej mieliło godzine
    def load_head(sheet, n=None):
        rows = []
        for i, row in enumerate(sheet.values):
            rows.append(row)
            if n is not None and i >= n:
                break
        return rows

    rows1 = load_head(sheet1, max_rows)
    rows2 = load_head(sheet2, max_rows)

    df_2009_2010 = pd.DataFrame(rows1[1:], columns=rows1[0])
    df_2010_2011 = pd.DataFrame(rows2[1:], columns=rows2[0])

    df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

    return df

def load_local() -> pd.DataFrame:

    # wczytanie danych z pliku
    file_path = "../data/online_retail_II.xlsx"
    df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

    # łączenie danych
    df = pd.concat([df1, df2], ignore_index=True)

    return df

