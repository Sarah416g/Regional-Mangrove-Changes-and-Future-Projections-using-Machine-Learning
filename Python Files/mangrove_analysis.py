"""
Name: Sarah Coulibaly
Course: CSE 163
Section: AA

Analyzes annual country-level mangrove extent from 1985 to 2025.

The program reads Mangrove_Extent.csv, which contains country ISO codes,
country names, and annual mangrove areas in hectares. It uses pandas
DataFrames to clean numeric values, calculate annual and total changes,
and summarize global and country-level trends. The program prints
statistics and saves six PNG graphs and summary CSV files to eda_outputs.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = "../data/Mangrove_Extent.csv"
OUTPUT_FOLDER = "eda_outputs"


def get_years(data: pd.DataFrame) -> list[str]:
    """
    Takes in a DataFrame (pd.DataFrame) containing country information
    and yearly mangrove extent. Selects and sorts the year columns and
    returns a list (list[str]) of year names in chronological order.
    """
    return sorted(data.columns[2:])


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame (pd.DataFrame) containing raw mangrove extent
    data. Removes rows without country identifiers, removes commas from
    numeric values, and converts yearly extents to numbers. Returns a
    cleaned DataFrame (pd.DataFrame) with a reset index. Invalid numeric
    values are converted to NaN.
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


def annual_changes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Calculates the difference in area between consecutive
    years for each country and returns a DataFrame (pd.DataFrame) of
    annual changes in hectares. The first year is NaN since there is
    no previous year to compare.
    """
    return data[get_years(data)].diff(axis=1)


def annual_variability(data: pd.DataFrame) -> pd.Series:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Calculates the standard deviation of annual changes
    for each country and returns a Series (pd.Series) of variability
    values in hectares indexed by country name.
    """
    variability = annual_changes(data).std(axis=1)
    variability.index = data["cnty_name"]
    return variability


def percent_changes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Calculates the annual percentage change for each
    country and returns a DataFrame (pd.DataFrame) of percentage changes.
    Undefined percentage changes, including division by zero, are NaN.
    """
    changes = data[get_years(data)].pct_change(
        axis=1, fill_method=None
    ) * 100
    return changes.replace([np.inf, -np.inf], np.nan)


def country_extremes(
    data: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Identifies the largest annual increase and decrease
    for each country and returns two Series (pd.Series) of changes in
    hectares indexed by country name.
    """
    changes = annual_changes(data)
    increases = changes.max(axis=1)
    decreases = changes.min(axis=1)
    increases.index = data["cnty_name"]
    decreases.index = data["cnty_name"]
    return increases, decreases


def net_changes(data: pd.DataFrame) -> pd.Series:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Subtracts the first year's area from the last year's
    area for each country and returns a Series (pd.Series) of net changes
    in hectares indexed by country name. Missing endpoint values are
    excluded.
    """
    years = get_years(data)
    changes = data[years[-1]] - data[years[0]]
    changes.index = data["cnty_name"]
    return changes.dropna()


def global_extent(data: pd.DataFrame) -> pd.Series:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned country-level
    mangrove extent data. Sums the area across countries for each year
    and returns a Series (pd.Series) of global mangrove extent in hectares.
    """
    return data[get_years(data)].sum(axis=0)


def global_gains_losses(
    data: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Calculates annual changes and separately sums gains
    and loss amounts across countries. Returns two Series (pd.Series)
    of annual gains and loss magnitudes in hectares, excluding the first
    year because there is no preceding year.
    """
    # Leave out 1985 because it has no earlier year for comparison.
    changes = annual_changes(data).iloc[:, 1:]
    gains = changes.clip(lower=0).sum(axis=0)
    losses = -changes.clip(upper=0).sum(axis=0)
    return gains, losses


def save_country_bar(
    values: pd.Series, title: str, ylabel: str, filename: str
) -> None:
    """
    Takes in a Series (pd.Series) of country values and three strings
    (str) for the graph title, y-axis label, and filename. Creates and
    saves a country bar chart as a PNG file and returns None. If the
    Series is empty, no chart is saved.
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


def save_plots(data: pd.DataFrame) -> None:
    """
    Takes in a DataFrame (pd.DataFrame) containing cleaned mangrove
    extent data. Creates six graphs showing global extent, annual gains
    and losses, the largest annual increases and decreases by country,
    and the largest net gains and losses by country. Saves the graphs
    as PNG files and returns None.
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
    plt.plot(gains.index, gains.values, label="Annual Gains", color="green")
    plt.plot(losses.index, losses.values, label="Annual Losses", color="red")
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
    # Display loss amounts as positive values so bars extend upward.
    biggest_losses = total_changes[total_changes < 0].nsmallest(10)
    save_country_bar(
        biggest_losses.abs(),
        "Top 10 Countries by Net Mangrove Loss (1985-2025)",
        "Net Mangrove Loss (ha)",
        "06_net_losses.png"
    )


def main() -> None:
    """
    Takes in no parameters and reads the mangrove extent CSV file.
    Cleans the data, calculates descriptive statistics and changes in
    mangrove area, and identifies countries with the greatest gains and
    losses. Prints the results, saves summary CSV files and six graphs,
    and returns None.
    """
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
