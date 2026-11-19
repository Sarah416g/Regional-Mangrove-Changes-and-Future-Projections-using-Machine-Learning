"""
Test the mangrove analysis functions with known example values.

The tests create small pandas DataFrames representing three fictional
countries, plus blank and summary rows. They check cleaning, changes,
variability, yearly totals, missing values, and country rankings using
assert statements. No external CSV file is needed for these tests.

Run this script from the Python Files folder with:
    python test_mangrove_analysis.py
"""

import pandas as pd

from mangrove_analysis import (clean_data, annual_changes, annual_variability,
                               percent_changes,
                               country_extremes, net_changes, global_extent,
                               global_gains_losses)


def sample_data() -> pd.DataFrame:
    """
    Takes in no parameters and creates a DataFrame (pd.DataFrame) of
    sample mangrove extent values for three fictional countries, a
    summary row, and a blank row. Returns the DataFrame for testing.
    """
    return pd.DataFrame({
        "iso": ["MGV", "BYL", "CST", None, None],
        "cnty_name": ["Mangrovia", "Bayland", "Coastaria", "Total (Ha)", None],
        "1985": ["1,000", "100", "0", "1,100", None],
        "1986": ["1,100", "80", "0", "1,180", None],
        "1987": ["1,050", "120", "5", "1,175", None]
    })


def test_cleaning() -> None:
    """
    Takes in no parameters and uses the sample DataFrame to test
    whether non-country rows are removed and numeric values are
    converted correctly. Checks expected results with assert statements
    and returns None. A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    assert len(data) == 3
    assert data.loc[0, "cnty_name"] == "Mangrovia"
    assert data.loc[0, "1985"] == 1000
    assert data.loc[1, "1986"] == 80


def test_annual_changes() -> None:
    """
    Takes in no parameters and tests annual change calculations using
    known values from the sample DataFrame. Checks increases, decreases,
    and the missing first-year change with assert statements and returns
    None. A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    changes = annual_changes(data)
    assert changes["1985"].isna().all()
    assert changes.loc[0, "1986"] == 100
    assert changes.loc[0, "1987"] == -50
    assert changes.loc[1, "1986"] == -20
    assert changes.loc[2, "1987"] == 5


def test_annual_variability() -> None:
    """
    Takes in no parameters and tests the standard deviation of annual
    mangrove changes using the sample DataFrame. Compares calculated
    values with expected results using assert statements and returns
    None. A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    variability = annual_variability(data)
    assert abs(variability["Mangrovia"] - 106.066) < 0.01
    assert abs(variability["Bayland"] - 42.426) < 0.01


def test_percent_changes() -> None:
    """
    Takes in no parameters and tests annual percentage changes using
    known values from the sample DataFrame. Checks expected percentages
    and division by zero with assert statements and returns None.
    A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    changes = percent_changes(data)
    assert abs(changes.loc[0, "1986"] - 10) < 0.001
    assert abs(changes.loc[1, "1986"] + 20) < 0.001
    assert pd.isna(changes.loc[2, "1987"])


def test_country_extremes() -> None:
    """
    Takes in no parameters and tests each fictional country's largest
    annual increase and decrease. Compares calculated results with known
    values using assert statements and returns None. A failed assertion
    raises an AssertionError.
    """
    data = clean_data(sample_data())
    increases, decreases = country_extremes(data)
    assert increases["Mangrovia"] == 100
    assert decreases["Mangrovia"] == -50
    assert increases["Bayland"] == 40
    assert decreases["Bayland"] == -20


def test_net_changes() -> None:
    """
    Takes in no parameters and tests net mangrove changes between the
    first and last years in the sample DataFrame. Compares changes with
    expected values using assert statements and returns None. A failed
    assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    changes = net_changes(data)
    assert changes["Mangrovia"] == 50
    assert changes["Bayland"] == 20
    assert changes["Coastaria"] == 5


def test_global_extent() -> None:
    """
    Takes in no parameters and tests the calculation of global
    mangrove extent using sample country values. Checks yearly totals
    without counting the summary row using assert statements and
    returns None. A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    totals = global_extent(data)
    assert totals["1985"] == 1100
    assert totals["1986"] == 1180
    assert totals["1987"] == 1175


def test_global_gains_losses() -> None:
    """
    Takes in no parameters and tests annual global gains and losses
    using the sample DataFrame. Compares summed gains and loss amounts
    with expected values using assert statements and returns None.
    A failed assertion raises an AssertionError.
    """
    data = clean_data(sample_data())
    gains, losses = global_gains_losses(data)
    assert gains["1986"] == 100
    assert losses["1986"] == 20
    assert gains["1987"] == 45
    assert losses["1987"] == 50


def test_missing_year() -> None:
    """
    Takes in no parameters and creates sample country data with a
    missing yearly value. Checks whether changes involving the missing
    year are NaN using assert statements and returns None. A failed
    assertion raises an AssertionError.
    """
    raw = pd.DataFrame({
        "iso": ["MGV"],
        "cnty_name": ["Mangrovia"],
        "1985": ["25"],
        "1986": [None],
        "1987": ["40"]
    })
    data = clean_data(raw)
    changes = annual_changes(data)
    assert pd.isna(changes.loc[0, "1986"])
    assert pd.isna(changes.loc[0, "1987"])


def main() -> None:
    """
    Takes in no parameters and runs all nine testing functions to
    verify mangrove analysis calculations. Prints a success message
    if all assertions pass and returns None. A failed assertion
    stops the program.
    """
    test_cleaning()
    test_annual_changes()
    test_annual_variability()
    test_percent_changes()
    test_country_extremes()
    test_net_changes()
    test_global_extent()
    test_global_gains_losses()
    test_missing_year()
    print("All 9 tests passed!")


if __name__ == "__main__":
    main()
