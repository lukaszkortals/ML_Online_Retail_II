from __future__ import annotations
import pandas as pd

from src.Online_Retail_II.constants import COL_INVOICE_NO, COL_CUSTOMER_ID, COL_QUANTITY, COL_UNIT_PRICE, \
    COL_INVOICE_DATE, COL_NUM_ITEMS, COL_TOTAL_PRICE, COL_COUNTRY, COL_YEAR, COL_MONTH, COL_DAY, COL_HOUR, COL_WEEKDAY, \
    COL_IS_WEEKEND, COL_INVOICE_TOTAL, COL_NUM_LINES, COL_AVG_UNIT_PRICE, COL_STOCK_CODE


def change_basic_features(df) -> pd.DataFrame:
    """
    na wierszach:
    - usuwanie korekt i wypełnianie pustych customer Id zerami - sprzedaż niezalogowanego klietna
    - konwersja InvoiceDate na datetime
    - kolumna TotalPrice (Quantity * Price)
    - usuwanie anulowanych faktur i zostawienie tylko UK
    - cechy czasowe (rok, miesiąc, dzień, godzina, dzień tygodnia, weekend)
    """
    df = df.copy()

    df = df[(df[COL_QUANTITY] > 0) & (df[COL_UNIT_PRICE] > 0)]

    df[COL_CUSTOMER_ID] = df[COL_CUSTOMER_ID].fillna(0).astype(int)

    df[COL_INVOICE_DATE] = pd.to_datetime(df[COL_INVOICE_DATE], errors="coerce")

    df[COL_TOTAL_PRICE] = df[COL_QUANTITY] * df[COL_UNIT_PRICE]

    df = df[~df[COL_INVOICE_NO].astype(str).str.startswith("C")]
    df = df[df[COL_COUNTRY] == "United Kingdom"]

    df[COL_YEAR] = df[COL_INVOICE_DATE].dt.year
    df[COL_MONTH] = df[COL_INVOICE_DATE].dt.month
    df[COL_DAY] = df[COL_INVOICE_DATE].dt.day
    df[COL_HOUR] = df[COL_INVOICE_DATE].dt.hour
    df[COL_WEEKDAY] = df[COL_INVOICE_DATE].dt.weekday  #0-6
    df[COL_IS_WEEKEND] = df[COL_WEEKDAY].isin([5, 6])

    return df

def build_invoice_level_features(df) -> pd.DataFrame:
    """
    agregowanie do (Invoice):
    - InvoiceTotal - suma TotalPrice
    - NumItems - suma Quantity
    - NumLines - liczba unikalnych StockCode
    - AvgUnitPrice - średnia Price
    - Country, Customer ID, InvoiceDate, IsCancelled - pierwsza wartość / max
    - ponownie cechy czasowe na poziomie faktury
    """
    df = change_basic_features(df)

    customer_col = COL_CUSTOMER_ID

    agg_dict = {
        COL_INVOICE_TOTAL: (COL_TOTAL_PRICE, "sum"),
        COL_NUM_ITEMS: (COL_QUANTITY, "sum"),
        COL_NUM_LINES: (COL_STOCK_CODE, "nunique"),
        COL_AVG_UNIT_PRICE: (COL_UNIT_PRICE, "mean"),
        COL_INVOICE_DATE: (COL_INVOICE_DATE, "first"),
        #COL_COUNTRY: (COL_COUNTRY, "first"),
    }

    if customer_col is not None:
        agg_dict[COL_CUSTOMER_ID] = (customer_col, "first")

    invoice_df = (
        df.groupby(COL_INVOICE_NO)
        .agg(**agg_dict)
        .reset_index()
    )

    # cechy czasowe faktury
    invoice_df[COL_YEAR] = invoice_df[COL_INVOICE_DATE].dt.year.astype(pd.CategoricalDtype(ordered=True))
    invoice_df[COL_MONTH] = invoice_df[COL_INVOICE_DATE].dt.month.astype(pd.CategoricalDtype(ordered=True))
    invoice_df[COL_DAY] = invoice_df[COL_INVOICE_DATE].dt.day.astype(pd.CategoricalDtype(ordered=True))
    invoice_df[COL_HOUR] = invoice_df[COL_INVOICE_DATE].dt.hour.astype(pd.CategoricalDtype(ordered=True))
    invoice_df[COL_WEEKDAY] = invoice_df[COL_INVOICE_DATE].dt.weekday.astype(pd.CategoricalDtype(ordered=True))
    invoice_df[COL_IS_WEEKEND] = invoice_df[COL_WEEKDAY].isin([5, 6])

    return invoice_df.drop(columns=[COL_INVOICE_NO, COL_INVOICE_DATE, COL_CUSTOMER_ID])

def prepare_ml_dataset(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_q_low: float = 0.02,
    outlier_q_high: float = 0.98,
) -> pd.DataFrame:
    """
    główny
    - wywołanie działania na wierszach - dodawanie uzupelnianie usuwanie
    - agreguje do poziomu Invoice
    - usuwanie outlierów
    """
    invoice_df = build_invoice_level_features(df)

    # outliery
    if remove_outliers:
        q_low, q_high = invoice_df[COL_INVOICE_TOTAL].quantile(
            [outlier_q_low, outlier_q_high]
        )
        invoice_df = invoice_df[
            invoice_df[COL_INVOICE_TOTAL].between(q_low, q_high)
        ]

    return invoice_df