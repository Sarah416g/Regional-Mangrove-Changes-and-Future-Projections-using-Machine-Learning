"""
Test mangrove calculations using small datasets with known results.

Run this file with: python test_mangrove_analysis.py
"""

import pandas as pd

from mangrove_analysis import (clean_data, annual_changes, annual_variability,
                               percent_changes,
                               country_extremes, net_changes, global_extent,
                               global_gains_losses)


def sample_data():
    """
    Create three countries and two non-country rows for testing.
    """
    return pd.DataFrame({
        "iso": ["MGV", "BYL", "CST", None, None],
        "cnty_name": ["Mangrovia", "Bayland", "Coastaria", "Total (Ha)", None],
        "1985": ["1,000", "100", "0", "1,100", None],
        "1986": ["1,100", "80", "0", "1,180", None],
        "1987": ["1,050", "120", "5", "1,175", None]
    })


def test_cleaning():
    """
    Check that summary rows are removed and comma values become numbers.
    """
    data = clean_data(sample_data())
    assert len(data) == 3
    assert data.loc[0, "cnty_name"] == "Mangrovia"
    assert data.loc[0, "1985"] == 1000
    assert data.loc[1, "1986"] == 80


def test_annual_changes():
    """
    Compare year-to-year changes with answers calculated by hand.
    """
    data = clean_data(sample_data())
    changes = annual_changes(data)
    assert changes["1985"].isna().all()
    assert changes.loc[0, "1986"] == 100
    assert changes.loc[0, "1987"] == -50
    assert changes.loc[1, "1986"] == -20
    assert changes.loc[2, "1987"] == 5


def test_annual_variability():
    """
    Check the spread of a country's year-to-year changes.
    """
    data = clean_data(sample_data())
    variability = annual_variability(data)
    assert abs(variability["Mangrovia"] - 106.066) < 0.01
    assert abs(variability["Bayland"] - 42.426) < 0.01


def test_percent_changes():
    """
    Check percentage changes and undefined changes from zero.
    """
    data = clean_data(sample_data())
    changes = percent_changes(data)
    assert abs(changes.loc[0, "1986"] - 10) < 0.001
    assert abs(changes.loc[1, "1986"] + 20) < 0.001
    assert pd.isna(changes.loc[2, "1987"])


def test_country_extremes():
    """
    Check the greatest one-year gains and losses for each country.
    """
    data = clean_data(sample_data())
    increases, decreases = country_extremes(data)
    assert increases["Mangrovia"] == 100
    assert decreases["Mangrovia"] == -50
    assert increases["Bayland"] == 40
    assert decreases["Bayland"] == -20


def test_net_changes():
    """
    Check that overall change is final year minus starting year.
    """
    data = clean_data(sample_data())
    changes = net_changes(data)
    assert changes["Mangrovia"] == 50
    assert changes["Bayland"] == 20
    assert changes["Coastaria"] == 5


def test_global_extent():
    """
    Check yearly totals without counting the summary row twice.
    """
    data = clean_data(sample_data())
    totals = global_extent(data)
    assert totals["1985"] == 1100
    assert totals["1986"] == 1180
    assert totals["1987"] == 1175


def test_global_gains_losses():
    """
    Check summed increases and decreases for consecutive years.
    """
    data = clean_data(sample_data())
    gains, losses = global_gains_losses(data)
    assert gains["1986"] == 100
    assert losses["1986"] == 20
    assert gains["1987"] == 45
    assert losses["1987"] == 50


def test_missing_year():
    """
    Make sure missing data does not create an incorrect annual change.
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


def main():
    """
    Run the tests and report whether they passed.
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
