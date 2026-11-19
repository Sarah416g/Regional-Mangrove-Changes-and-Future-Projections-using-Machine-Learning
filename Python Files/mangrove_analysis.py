"""
Analyze country-level mangrove area from 1985 through 2025.

Run this file from the project folder with: python mangrove_analysis.py
Graphs and tables are saved in the eda_outputs folder.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = "../data/Mangrove_Extent.csv"
OUTPUT_FOLDER = "eda_outputs"


def get_years(data):
    """
    Return the year columns, ordered from earliest to latest.
    """
    return sorted(data.columns[2:])


def clean_data(data):
    """
    Keep country rows and convert mangrove areas to numbers.
    """
    # Rows without an ISO code are blank or summary rows, not countries.
    countries = data.dropna(subset=["iso", "cnty_name"]).copy()
    years = get_years(countries)

    # Remove commas from values like "26,457" before calculations.
    # Apply the same cleaning step to every year column.
    countries[years] = countries[years].astype(str).apply(
        lambda column: column.str.replace(",", "")
    )
    countries[years] = countries[years].apply(
        pd.to_numeric, errors="coerce"
    )

    return countries.reset_index(drop=True)


def annual_changes(data):
    """
    Calculate each country's change from the previous year in hectares.
    """
    return data[get_years(data)].diff(axis=1)


def annual_variability(data):
    """
    Measure how much each country's annual changes vary over time.
    """
    variability = annual_changes(data).std(axis=1)
    variability.index = data["cnty_name"]
    return variability


def percent_changes(data):
    """
    Calculate annual percentage changes, keeping undefined values empty.
    """
    changes = data[get_years(data)].pct_change(
        axis=1, fill_method=None
    ) * 100
    return changes.replace([np.inf, -np.inf], np.nan)


def country_extremes(data):
    """
    Find each country's largest one-year increase and decrease.
    """
    changes = annual_changes(data)
    increases = changes.max(axis=1)
    decreases = changes.min(axis=1)
    increases.index = data["cnty_name"]
    decreases.index = data["cnty_name"]
    return increases, decreases


def net_changes(data):
    """
    Calculate each country's change between 1985 and 2025.
    """
    years = get_years(data)
    changes = data[years[-1]] - data[years[0]]
    changes.index = data["cnty_name"]
    return changes.dropna()


def global_extent(data):
    """
    Sum the mangrove area of all countries for each year.
    """
    return data[get_years(data)].sum(axis=0)


def global_gains_losses(data):
    """
    Total positive and negative country changes for each year.
    """
    # Leave out 1985 because it has no earlier year for comparison.
    changes = annual_changes(data).iloc[:, 1:]
    gains = changes.clip(lower=0).sum(axis=0)
    losses = -changes.clip(upper=0).sum(axis=0)
    return gains, losses


def save_country_bar(values, title, ylabel, filename):
    """
    Save a bar graph comparing up to ten countries.
    """
    if values.empty:
        return

    plt.figure(figsize=(11, 6))
    values.plot(kind="bar")
    plt.xlabel("Country")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=200)
    plt.close()


def save_plots(data):
    """
    Save six graphs showing global trends and country-level changes.

    The graphs show global extent, annual gains and losses, largest one-year
    increases, largest one-year decreases, net gains, and net losses.
    """
    years = get_years(data)
    totals = global_extent(data)
    gains, losses = global_gains_losses(data)
    increases, decreases = country_extremes(data)
    total_changes = net_changes(data)

    # 1. Global mangrove area in hectares for each year from 1985 to 2025.
    plt.figure(figsize=(11, 5))
    plt.plot(years, totals.values)
    plt.xlabel("Year")
    plt.ylabel("Total Mangrove Extent (ha)")
    plt.title("Global Mangrove Extent Over Time (1985-2025)")
    plt.xticks(years[::5], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "01_global_extent.png"),
                dpi=200)
    plt.close()

    # 2. Compare yearly increases and decreases across all countries.
    # Both lines show positive magnitudes to make them easier to compare.
    plt.figure(figsize=(11, 5))
    plt.plot(gains.index, gains.values, label="Annual Gains")
    plt.plot(losses.index, losses.values, label="Annual Losses")
    plt.xlabel("Year")
    plt.ylabel("Annual Mangrove Extent Change (ha)")
    plt.title("Global Annual Mangrove Gains and Losses (1986-2025)")
    plt.xticks(gains.index[::5], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "02_global_gains_losses.png"),
                dpi=200)
    plt.close()

    # 3. Countries with the largest increase in any one year.
    biggest_increases = increases[increases > 0].nlargest(10)
    save_country_bar(
        biggest_increases,
        "Top 10 Countries by Largest Annual Mangrove Increase",
        "Largest Annual Increase (ha)",
        "03_annual_increases.png"
    )

    # 4. Countries with the largest decrease in any one year.
    # Show loss amounts as positive values, as in the country bar graph.
    biggest_decreases = decreases[decreases < 0].nsmallest(10)
    save_country_bar(
        biggest_decreases.abs(),
        "Top 10 Countries by Largest Annual Mangrove Decrease",
        "Largest Annual Loss (ha)",
        "04_annual_decreases.png"
    )

    # 5. Countries with the greatest overall gains from 1985 to 2025.
    biggest_gains = total_changes[total_changes > 0].nlargest(10)
    save_country_bar(
        biggest_gains,
        "Top 10 Countries by Net Mangrove Gain (1985-2025)",
        "Net Change in Mangrove Extent (ha)",
        "05_net_gains.png"
    )

    # 6. Countries with the greatest overall losses from 1985 to 2025.
    # Keep losses negative because this graph shows net change.
    biggest_losses = total_changes[total_changes < 0].nsmallest(10)
    save_country_bar(
        biggest_losses,
        "Top 10 Countries by Net Mangrove Loss (1985-2025)",
        "Net Change in Mangrove Extent (ha)",
        "06_net_losses.png"
    )


def main():
    """Load the data, print summary statistics, and save graphs and tables."""
    raw = pd.read_csv(DATA_FILE)
    data = clean_data(raw)
    years = get_years(data)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    differences = annual_changes(data)
    percent = percent_changes(data)
    variability = annual_variability(data)
    increases, decreases = country_extremes(data)
    country_changes = net_changes(data)
    totals = global_extent(data)
    gains, losses = global_gains_losses(data)

    # Dataset size and missing values are important for interpreting results.
    print("Original rows and columns:", raw.shape)
    print("Country-only rows and columns:", data.shape)
    print("\nMissing values in original data:")
    print(raw.isna().sum())
    print("\nMissing mangrove areas after cleaning:")
    print(data[years].isna().sum())

    # describe() gives count, mean, std, min, Q1, median, Q3, and max.
    print("\nMangrove area summary for 1985 and 2025:")
    print(data[[years[0], years[-1]]].describe())
    print("\nLargest annual increase summary:")
    print(increases.describe())
    print("\nLargest annual decrease summary:")
    print(decreases.describe())
    print("\nAnnual variability summary:")
    print(variability.describe())
    print("\nNet change summary:")
    print(country_changes.describe())

    # Small starting areas can produce large percent changes.
    largest_percent_increases = percent.max(axis=1)
    largest_percent_decreases = percent.min(axis=1)
    print("\nLargest annual percentage increase summary:")
    print(largest_percent_increases.describe())
    print("\nLargest annual percentage decrease summary:")
    print(largest_percent_decreases.describe())
    largest_percent_increases.index = data["cnty_name"]
    largest_percent_decreases.index = data["cnty_name"]
    print("Largest percentage gain:", largest_percent_increases.idxmax())
    print("Largest percentage loss:", largest_percent_decreases.idxmin())

    print("\nGlobal extent in the first and last year:")
    print(totals.iloc[0], totals.iloc[-1])
    print("Year with the most annual gains:", gains.idxmax())
    print("Year with the most annual losses:", losses.idxmax())
    print("\nTop 10 countries by net gain:")
    print(country_changes[country_changes > 0].nlargest(10))
    print("\nTop 10 countries by net loss:")
    print(country_changes[country_changes < 0].nsmallest(10))
    print("\nCountries with net gains:", (country_changes > 0).sum())
    print("Countries with net losses:", (country_changes < 0).sum())
    print("Countries with no net change:", (country_changes == 0).sum())

    # Save the processed data and numbers used to make the plots.
    data.to_csv(os.path.join(OUTPUT_FOLDER, "cleaned_extent.csv"),
                index=False)
    data[years].describe().to_csv(
        os.path.join(OUTPUT_FOLDER, "extent_summaries.csv")
    )
    differences.to_csv(os.path.join(OUTPUT_FOLDER, "annual_changes.csv"),
                       index=False)
    percent.to_csv(os.path.join(OUTPUT_FOLDER, "percent_changes.csv"),
                   index=False)
    country_changes.to_csv(os.path.join(OUTPUT_FOLDER, "net_changes.csv"))

    save_plots(data)
    print("\nSaved graphs and tables in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
