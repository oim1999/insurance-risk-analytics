"""
hypothesis_tests.py
Reusable statistical hypothesis testing functions for ACIS insurance data.

Each function follows the same pattern:
1. Segment data into Group A (control) and Group B (test)
2. Select appropriate statistical test based on KPI type
3. Calculate test statistic and p-value
4. Return results in a standardized dictionary
"""
import pandas as pd
import numpy as np
from scipy import stats


def test_risk_difference_provinces(df, province_a, province_b, kpi='claim_frequency'):
    """
    Test H0: No risk difference between two provinces.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned insurance data
    province_a : str
        Control province (baseline)
    province_b : str
        Test province (comparison)
    kpi : str
        'claim_frequency' (chi-squared) or 'claim_severity' (t-test)

    Returns:
    --------
    dict with test results
    """
    # Filter to the two provinces
    group_a = df[df['province'] == province_a].copy()
    group_b = df[df['province'] == province_b].copy()

    results = {
        'hypothesis': f'No risk difference between {province_a} and {province_b}',
        'group_a': province_a,
        'group_b': province_b,
        'kpi': kpi,
        'group_a_n': len(group_a),
        'group_b_n': len(group_b)
    }

    if kpi == 'claim_frequency':
        # Claim frequency = proportion with claims > 0
        # Chi-squared test on contingency table
        a_claims = (group_a['totalclaims'] > 0).sum()
        a_no_claims = len(group_a) - a_claims
        b_claims = (group_b['totalclaims'] > 0).sum()
        b_no_claims = len(group_b) - b_claims

        contingency = np.array([[a_claims, a_no_claims],
                                [b_claims, b_no_claims]])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        results['test'] = 'Chi-squared'
        results['statistic'] = chi2
        results['p_value'] = p_value
        results['group_a_freq'] = a_claims / len(group_a) * 100
        results['group_b_freq'] = b_claims / len(group_b) * 100

    elif kpi == 'claim_severity':
        # Claim severity = average claim amount (only policies with claims)
        a_severe = group_a[group_a['totalclaims'] > 0]['totalclaims']
        b_severe = group_b[group_b['totalclaims'] > 0]['totalclaims']

        # Welch's t-test (does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(a_severe, b_severe, equal_var=False)

        results['test'] = 'Welch t-test'
        results['statistic'] = t_stat
        results['p_value'] = p_value
        results['group_a_mean'] = a_severe.mean()
        results['group_b_mean'] = b_severe.mean()

    else:
        raise ValueError("kpi must be 'claim_frequency' or 'claim_severity'")

    results['decision'] = 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
    results['significant'] = p_value < 0.05

    return results


def test_risk_difference_zipcodes(df, zip_a, zip_b, kpi='claim_frequency'):
    """
    Test H0: No risk difference between two zip codes.
    Same structure as provinces but for postal codes.
    """
    group_a = df[df['postalcode'] == zip_a].copy()
    group_b = df[df['postalcode'] == zip_b].copy()

    # Skip if either group is too small
    if len(group_a) < 30 or len(group_b) < 30:
        return {
            'hypothesis': f'No risk difference between zip {zip_a} and {zip_b}',
            'error': 'Insufficient sample size (need >= 30 per group)',
            'group_a_n': len(group_a),
            'group_b_n': len(group_b)
        }

    results = {
        'hypothesis': f'No risk difference between zip {zip_a} and {zip_b}',
        'group_a': str(zip_a),
        'group_b': str(zip_b),
        'kpi': kpi,
        'group_a_n': len(group_a),
        'group_b_n': len(group_b)
    }

    if kpi == 'claim_frequency':
        a_claims = (group_a['totalclaims'] > 0).sum()
        a_no_claims = len(group_a) - a_claims
        b_claims = (group_b['totalclaims'] > 0).sum()
        b_no_claims = len(group_b) - b_claims

        contingency = np.array([[a_claims, a_no_claims],
                                [b_claims, b_no_claims]])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        results['test'] = 'Chi-squared'
        results['statistic'] = chi2
        results['p_value'] = p_value
        results['group_a_freq'] = a_claims / len(group_a) * 100
        results['group_b_freq'] = b_claims / len(group_b) * 100

    elif kpi == 'claim_severity':
        a_severe = group_a[group_a['totalclaims'] > 0]['totalclaims']
        b_severe = group_b[group_b['totalclaims'] > 0]['totalclaims']

        t_stat, p_value = stats.ttest_ind(a_severe, b_severe, equal_var=False)

        results['test'] = 'Welch t-test'
        results['statistic'] = t_stat
        results['p_value'] = p_value
        results['group_a_mean'] = a_severe.mean()
        results['group_b_mean'] = b_severe.mean()

    results['decision'] = 'Reject H0' if results.get('p_value', 1) < 0.05 else 'Fail to reject H0'
    results['significant'] = results.get('p_value', 1) < 0.05

    return results


def test_margin_difference_zipcodes(df, zip_a, zip_b):
    """
    Test H0: No margin difference between two zip codes.
    Margin is numerical, so we use t-test.
    """
    group_a = df[df['postalcode'] == zip_a]['margin']
    group_b = df[df['postalcode'] == zip_b]['margin']

    if len(group_a) < 30 or len(group_b) < 30:
        return {
            'hypothesis': f'No margin difference between zip {zip_a} and {zip_b}',
            'error': 'Insufficient sample size',
            'group_a_n': len(group_a),
            'group_b_n': len(group_b)
        }

    # Welch's t-test on margin
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    return {
        'hypothesis': f'No margin difference between zip {zip_a} and {zip_b}',
        'group_a': str(zip_a),
        'group_b': str(zip_b),
        'kpi': 'margin',
        'test': 'Welch t-test',
        'statistic': t_stat,
        'p_value': p_value,
        'group_a_n': len(group_a),
        'group_b_n': len(group_b),
        'group_a_mean': group_a.mean(),
        'group_b_mean': group_b.mean(),
        'decision': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0',
        'significant': p_value < 0.05
    }


def test_risk_difference_gender(df, gender_a='Female', gender_b='Male', kpi='claim_frequency'):
    """
    Test H0: No risk difference between women and men.

    Note: We exclude 'Not Specified' to ensure clean comparison.
    """
    # Filter to the two genders only
    df_gender = df[df['gender'].isin([gender_a, gender_b])].copy()

    group_a = df_gender[df_gender['gender'] == gender_a].copy()
    group_b = df_gender[df_gender['gender'] == gender_b].copy()

    results = {
        'hypothesis': f'No risk difference between {gender_a} and {gender_b}',
        'group_a': gender_a,
        'group_b': gender_b,
        'kpi': kpi,
        'group_a_n': len(group_a),
        'group_b_n': len(group_b)
    }

    if kpi == 'claim_frequency':
        a_claims = (group_a['totalclaims'] > 0).sum()
        a_no_claims = len(group_a) - a_claims
        b_claims = (group_b['totalclaims'] > 0).sum()
        b_no_claims = len(group_b) - b_claims

        contingency = np.array([[a_claims, a_no_claims],
                                [b_claims, b_no_claims]])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        results['test'] = 'Chi-squared'
        results['statistic'] = chi2
        results['p_value'] = p_value
        results['group_a_freq'] = a_claims / len(group_a) * 100
        results['group_b_freq'] = b_claims / len(group_b) * 100

    elif kpi == 'claim_severity':
        a_severe = group_a[group_a['totalclaims'] > 0]['totalclaims']
        b_severe = group_b[group_b['totalclaims'] > 0]['totalclaims']

        t_stat, p_value = stats.ttest_ind(a_severe, b_severe, equal_var=False)

        results['test'] = 'Welch t-test'
        results['statistic'] = t_stat
        results['p_value'] = p_value
        results['group_a_mean'] = a_severe.mean()
        results['group_b_mean'] = b_severe.mean()

    results['decision'] = 'Reject H0' if results.get('p_value', 1) < 0.05 else 'Fail to reject H0'
    results['significant'] = results.get('p_value', 1) < 0.05

    return results


def print_test_results(results):
    """
    Pretty-print test results for notebook display.
    """
    print("=" * 60)
    print(f"HYPOTHESIS: {results['hypothesis']}")
    print("=" * 60)

    if 'error' in results:
        print(f"ERROR: {results['error']}")
        print(f"Group A n: {results.get('group_a_n', 'N/A')}")
        print(f"Group B n: {results.get('group_b_n', 'N/A')}")
        return

    print(f"KPI: {results['kpi']}")
    print(f"Test: {results['test']}")
    print(f"Group A ({results['group_a']}): n = {results['group_a_n']:,}")
    print(f"Group B ({results['group_b']}): n = {results['group_b_n']:,}")
    print(f"Statistic: {results['statistic']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")

    if 'group_a_freq' in results:
        print(f"Group A claim frequency: {results['group_a_freq']:.2f}%")
        print(f"Group B claim frequency: {results['group_b_freq']:.2f}%")

    if 'group_a_mean' in results:
        print(f"Group A mean: R{results['group_a_mean']:,.2f}")
        print(f"Group B mean: R{results['group_b_mean']:,.2f}")

    print(f"DECISION: {results['decision']}")

    if results['significant']:
        print(">>> INTERPRETATION: Statistically significant difference detected.")
    else:
        print(">>> INTERPRETATION: No statistically significant difference.")


def create_results_table(all_results):
    """
    Create a summary DataFrame from multiple test results.

    Parameters:
    -----------
    all_results : list of dict
        Output from test functions

    Returns:
    --------
    pd.DataFrame
    """
    rows = []
    for r in all_results:
        if 'error' in r:
            continue
        rows.append({
            'Hypothesis': r['hypothesis'],
            'KPI': r['kpi'],
            'Test': r['test'],
            'Group A': r['group_a'],
            'Group B': r['group_b'],
            'Statistic': f"{r['statistic']:.4f}",
            'P-Value': f"{r['p_value']:.6f}",
            'Decision': r['decision']
        })

    return pd.DataFrame(rows)


def find_extreme_groups(df, feature, metric='loss_ratio', min_n=1000):
    """
    Helper to find the two most extreme groups for A/B testing.

    Parameters:
    -----------
    df : pd.DataFrame
    feature : str
        Column to group by (e.g., 'province', 'postalcode')
    metric : str
        Metric to rank by
    min_n : int
        Minimum group size to consider

    Returns:
    --------
    tuple (lowest_group, highest_group)
    """
    grouped = df.groupby(feature).agg(
        total_premium=('totalpremium', 'sum'),
        total_claims=('totalclaims', 'sum'),
        count=('totalpremium', 'count')
    ).reset_index()

    grouped['loss_ratio'] = grouped['total_claims'] / grouped['total_premium']
    grouped = grouped[grouped['count'] >= min_n]
    grouped = grouped.sort_values('loss_ratio')

    lowest = grouped.iloc[0][feature]
    highest = grouped.iloc[-1][feature]

    print(f"Lowest {feature}: {lowest} (LR: {grouped.iloc[0]['loss_ratio']:.4f}, n: {grouped.iloc[0]['count']:,})")
    print(f"Highest {feature}: {highest} (LR: {grouped.iloc[-1]['loss_ratio']:.4f}, n: {grouped.iloc[-1]['count']:,})")

    return lowest, highest
