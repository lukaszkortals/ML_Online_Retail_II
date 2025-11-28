# Projekt ML – Online Retail II

Projekt realizowany w ramach studiów podyplomowych **„Sztuczna Inteligencja i automatyzacja procesów biznesowych”**.  
Celem projektu jest zbudowanie i zinterpretowanie modelu regresyjnego przewidującego **wartość faktury (InvoiceTotal)** na podstawie danych transakcyjnych z zestawu **Online Retail II**.

---

## 1. Cel projektu

- przygotowanie kompletnego pipeline’u ML dla problemu regresji,
- porównanie kilku modeli (benchmark, modele liniowe, modele drzewiaste),
- optymalizacja hiperparametrów (Optuna),
- interpretacja modelu (SHAP, feature importance, permutation importance),
- wyciągnięcie wniosków na podstawie wyników.

---

## 2. Dane

W projekcie wykorzystano zbiór **Online Retail II** z Kaggle:

- dane transakcyjne ze sklepu internetowego,
- faktury, produkty, ilości, ceny jednostkowe, klient, kraj, data,
- dwa arkusze: `Year 2009-2010` oraz `Year 2010-2011`,
- w projekcie agregujemy dane do poziomu **pojedynczej faktury**.

Główne kroki przygotowania danych:

1. Pobranie datasetu (np. przez `kagglehub` lub ręcznie).
2. Połączenie obu arkuszy w jeden DataFrame.
3. Czyszczenie danych:
   - usunięcie faktur anulowanych (`Invoice` zaczyna się od `C`),
   - usunięcie rekordów z `Quantity <= 0` lub `Price <= 0`,
   - filtrowanie tylko transakcji z `United Kingdom`,
   - dodatkowo po features engineering'u usunięcie kolumn `CustomerID`, `InvoiceDate`,`InvoiceNo`, `Country`
4. Utworzenie cech:
   - `TotalPrice` = `Quantity * Price`,
   - cechy czasowe na podstawie `InvoiceDate` (rok, miesiąc, dzień tygodnia, itp.),
   - agregacja do poziomu faktury (sumy, średnie, liczba pozycji).

---

## 3. Struktura projektu

Przykładowa struktura katalogów:

```text
ML_Online_Retail_II/
├── notebooks/
│   └── project.ipynb        # główny notebook z prezentacją całego pipeline’u
├── src/
│   └── Online_Retail_II/
│       ├── __init__.py
│       ├── constants.py     # nazwy kolumn jako stałe
│       ├── data.py          # ładowanie i podstawowe czyszczenie danych
│       ├── features.py      # feature engineering i agregacja do faktury
│       ├── models.py        # definicje modeli, pipeline’y, Optuna
│       └── interpretation.py# SHAP, feature importance, permutation importance, Lasso
├── requirements.txt         # zależności projektu
├── environment.yml          # konfiguracja środowiska wirtualnego
└── README.md                # ten plik
```
## 4. Technologie
- Python 3.11+
- pandas, numpy
- scikit-learn
- optuna
- xgboost (opcjonalnie)
- shap
- matplotlib, seaborn
- kagglehub (do pobrania danych z Kaggle)

## 5. Jak uruchomić projekt

### 5.1 Utworzenie środowiska
```bash
conda env create -f environment.yml
conda activate py3139
```

## 6. Wnioski

- Największy wpływ na wartość faktury mają: liczba pozycji, liczba sztuk oraz średnia cena.  
- SHAP pokazuje wyraźnie, że zależności są nieliniowe – dlatego modele drzewiaste działają najlepiej.  
- Dane mają silną sezonowość i niejednorodny rozkład, dlatego użyto trimming percentylowy oraz agregacji do poziomu faktury.

Pełna interpretacja (SHAP, Lasso, permutation importance) znajduje się w notebooku.