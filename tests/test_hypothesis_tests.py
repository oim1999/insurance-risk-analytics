"""
test_hypothesis_tests.py
Unit tests for hypothesis testing functions.
"""
import pandas as pd
import numpy as np
import pytest
from src.hypothesis_tests import (
    test_risk_difference_gender,
    test_risk_difference_provinces
)


def test_gender_test_structure():
    """Test that gender test returns correct dictionary keys."""
    # Create minimal test data
    df = pd.DataFrame({
        'gender': ['Female', 'Female', 'Male', 'Male', 'Not Specified'],
        'totalclaims': [0, 100, 0, 200, 0],
        'totalpremium': [100, 100, 100, 100, 100],
        'margin': [100, 0, 100, -100, 100]
    })

    result = test_risk_difference_gender(df, 'Female', 'Male', 'claim_frequency')

    assert 'hypothesis' in result
    assert 'p_value' in result
    assert 'decision' in result
    assert result['group_a'] == 'Female'
    assert result['group_b'] == 'Male'


def test_province_test_with_small_data():
    """Test province test with minimal data."""
    df = pd.DataFrame({
        'province': ['A', 'A', 'B', 'B'],
        'totalclaims': [0, 50, 0, 100],
        'totalpremium': [100, 100, 100, 100]
    })

    result = test_risk_difference_provinces(df, 'A', 'B', 'claim_frequency')

    assert result['group_a'] == 'A'
    assert result['group_b'] == 'B'
    assert 'p_value' in result
