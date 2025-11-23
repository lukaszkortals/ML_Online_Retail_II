from __future__ import annotations
import pandas as pd
from typing import Tuple

from src.Online_Retail_II.constants import COL_INVOICE_NO


def change_basic_features(df) -> pd.DataFrame:
    """
    Zmiany na wierszach:
    - usuwanie korekt i wypełnianie pustych customer Id zerami - sprzedaż niezalogowanego klietna
    - konwersja InvoiceDate na datetime
    - kolumna TotalPrice (Quantity * Price)
    - usuwanie anulowanych faktur i zostawienie tylko UK
    - cechy czasowe (rok, miesiąc, dzień, godzina, dzień tygodnia, weekend)
    """
    df = df.copy()

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    df["Customer ID"] = df["Customer ID"].fillna(0).astype(int)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df["TotalPrice"] = df["Quantity"] * df["Price"]

    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Country"] == "United Kingdom"]

    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"] = df["InvoiceDate"].dt.day
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour
    df["InvoiceWeekday"] = df["InvoiceDate"].dt.weekday  #0-6
    df["IsWeekend"] = df["InvoiceWeekday"].isin([5, 6])

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

    customer_col = "Customer ID"

    agg_dict = {
        "InvoiceTotal": ("TotalPrice", "sum"),
        "NumItems": ("Quantity", "sum"),
        "NumLines": ("StockCode", "nunique"),
        "AvgUnitPrice": ("Price", "mean"),
        "InvoiceDate": ("InvoiceDate", "first"),
        "Country": ("Country", "first"),
    }

    if customer_col is not None:
        agg_dict["Customer ID"] = (customer_col, "first")

    invoice_df = (
        df.groupby(COL_INVOICE_NO)
        .agg(**agg_dict)
        .reset_index()
    )

    # Cechy czasowe już na poziomie faktury
    invoice_df["InvoiceYear"] = invoice_df["InvoiceDate"].dt.year
    invoice_df["InvoiceMonth"] = invoice_df["InvoiceDate"].dt.month
    invoice_df["InvoiceDay"] = invoice_df["InvoiceDate"].dt.day
    invoice_df["InvoiceHour"] = invoice_df["InvoiceDate"].dt.hour
    invoice_df["InvoiceWeekday"] = invoice_df["InvoiceDate"].dt.weekday
    invoice_df["IsWeekend"] = invoice_df["InvoiceWeekday"].isin([5, 6])

    return invoice_df

def prepare_ml_dataset(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_q_low: float = 0.01,
    outlier_q_high: float = 0.99,
) -> pd.DataFrame:
    """
    Finalny dataset pod model ML (regresja InvoiceTotal na poziomie faktury).
    - wywołanie działania na wierszach - dodawanie uzupelnianie usuwanie
    - agreguje do poziomu Invoice
    - usuwanie outlierów
    """
    invoice_df = build_invoice_level_features(df)

    # outliery
    if remove_outliers:
        q_low, q_high = invoice_df["InvoiceTotal"].quantile(
            [outlier_q_low, outlier_q_high]
        )
        invoice_df = invoice_df[
            invoice_df["InvoiceTotal"].between(q_low, q_high)
        ]

    return invoice_df