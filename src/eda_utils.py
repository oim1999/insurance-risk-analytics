"""
eda_utils.py
Reusable plotting and analysis functions for insurance EDA.
Each function is self-contained and returns the figure for display/saving.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_univariate_numerical(df, column, bins=50, figsize=(10, 5)):
    """
    Plot a histogram with KDE for any numerical column.
    Also shows mean and median lines.

    Why: Histograms show distribution shape (normal, skewed, bimodal).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Drop missing values for plotting
    data = df[column].dropna()

    # Histogram + KDE
    sns.histplot(data, bins=bins, kde=True, ax=ax, color='steelblue', alpha=0.7)

    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:,.2f}')
    ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:,.2f}')

    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_univariate_categorical(df, column, top_n=15, figsize=(12, 6)):
    """
    Plot a bar chart for categorical columns.
    Shows top N categories by frequency.

    Why: Bar charts make it easy to see which categories dominate.
    """
    fig, ax = plt.subplots(figsize=figsize)

    value_counts = df[column].value_counts().head(top_n)

    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Categories in {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel(column, fontsize=12)

    # Add count labels on bars
    for i, v in enumerate(value_counts.values):
        ax.text(v + max(value_counts.values)*0.01, i, str(v), va='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_boxplot_outliers(df, columns, figsize=(14, 8)):
    """
    Create box plots for multiple numerical columns to detect outliers.

    Why: Box plots show quartiles, median, and outliers (points beyond 1.5*IQR).
    Outliers can skew means and ruin models.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=figsize)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.boxplot(y=df[col], ax=ax, color='lightcoral')
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_ylabel('')

    fig.suptitle('Outlier Detection via Box Plots', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df, columns=None, figsize=(12, 10)):
    """
    Plot a heatmap of correlations between numerical columns.

    Why: Correlation shows which variables move together.
    High correlation between features can cause multicollinearity in models.
    """
    if columns is None:
        # Select only numerical columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_premium_vs_claims_by_group(df, group_col, sample_size=5000, figsize=(14, 8)):
    """
    Scatter plot of TotalPremium vs TotalClaims, colored by a categorical group.

    Why: This reveals if certain groups (e.g., provinces) cluster in high-risk zones.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sample for performance if dataset is large
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42)
    else:
        plot_df = df

    # Get unique groups and assign colors
    groups = plot_df[group_col].dropna().unique()
    colors = sns.color_palette("tab10", len(groups))

    for group, color in zip(groups, colors):
        subset = plot_df[plot_df[group_col] == group]
        ax.scatter(subset['totalpremium'], subset['totalclaims'], 
                  alpha=0.5, s=20, label=str(group), color=color)

    ax.set_xlabel('Total Premium (R)', fontsize=12)
    ax.set_ylabel('Total Claims (R)', fontsize=12)
    ax.set_title(f'Total Premium vs Total Claims by {group_col}', fontsize=14, fontweight='bold')
    ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add diagonal line where Claims = Premium (Loss Ratio = 1)
    max_val = max(plot_df['totalpremium'].max(), plot_df['totalclaims'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Loss Ratio = 1.0')

    plt.tight_layout()
    return fig


def plot_loss_ratio_by_category(df, category_col, min_policies=50, figsize=(14, 8)):
    """
    Calculate and plot average Loss Ratio by a categorical variable.
    Only includes categories with at least min_policies to avoid noise.

    Why: This directly answers "how does loss ratio vary by X?"
    """
    # Group and calculate metrics
    grouped = df.groupby(category_col).agg({
        'loss_ratio': 'mean',
        'totalpremium': 'sum',
        'totalclaims': 'sum'
    }).reset_index()

    # Count policies per category
    counts = df[category_col].value_counts().reset_index()
    counts.columns = [category_col, 'policy_count']
    grouped = grouped.merge(counts, on=category_col)

    # Filter for minimum policy count
    grouped = grouped[grouped['policy_count'] >= min_policies]
    grouped = grouped.sort_values('loss_ratio', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['green' if x < 0.5 else 'orange' if x < 1.0 else 'red' for x in grouped['loss_ratio']]

    ax.barh(grouped[category_col].astype(str), grouped['loss_ratio'], color=colors, alpha=0.8)
    ax.axvline(0.5, color='blue', linestyle='--', alpha=0.7, label='Target (0.5)')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven (1.0)')
    ax.set_xlabel('Average Loss Ratio', fontsize=12)
    ax.set_ylabel(category_col, fontsize=12)
    ax.set_title(f'Average Loss Ratio by {category_col}', fontsize=14, fontweight='bold')
    ax.legend()

    # Add value labels
    for i, v in enumerate(grouped['loss_ratio']):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_geographic_trends(df, figsize=(16, 10)):
    """
    Create a multi-panel plot showing geographic trends:
    - Average Premium by Province
    - Average Claims by Province
    - Loss Ratio by Province
    - Top Vehicle Makes by Province

    Why: Geographic segmentation is key for regional marketing strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Average Premium by Province
    prov_premium = df.groupby('province')['totalpremium'].mean().sort_values(ascending=False)
    sns.barplot(x=prov_premium.values, y=prov_premium.index, ax=axes[0,0], palette='Blues_d')
    axes[0,0].set_title('Average Premium by Province', fontweight='bold')
    axes[0,0].set_xlabel('Avg Premium (R)')

    # 2. Average Claims by Province
    prov_claims = df.groupby('province')['totalclaims'].mean().sort_values(ascending=False)
    sns.barplot(x=prov_claims.values, y=prov_claims.index, ax=axes[0,1], palette='Reds_d')
    axes[0,1].set_title('Average Claims by Province', fontweight='bold')
    axes[0,1].set_xlabel('Avg Claims (R)')

    # 3. Loss Ratio by Province
    prov_lr = df.groupby('province')['loss_ratio'].mean().sort_values(ascending=False)
    colors = ['green' if x < 0.5 else 'orange' if x < 1.0 else 'red' for x in prov_lr.values]
    sns.barplot(x=prov_lr.values, y=prov_lr.index, ax=axes[1,0], palette=colors)
    axes[1,0].set_title('Loss Ratio by Province', fontweight='bold')
    axes[1,0].set_xlabel('Loss Ratio')
    axes[1,0].axvline(1.0, color='black', linestyle='--', alpha=0.5)

    # 4. Top Vehicle Makes (overall)
    top_makes = df['make'].value_counts().head(10)
    sns.barplot(x=top_makes.values, y=top_makes.index, ax=axes[1,1], palette='viridis')
    axes[1,1].set_title('Top 10 Vehicle Makes (Portfolio)', fontweight='bold')
    axes[1,1].set_xlabel('Count')

    fig.suptitle('Geographic and Vehicle Trends', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_temporal_trends(df, figsize=(14, 10)):
    """
    Analyze trends over the 18-month period.
    Shows: monthly premium, claims, claim frequency, and average claim severity.

    Why: Temporal trends reveal seasonality or deteriorating portfolio performance.
    """
    df = df.copy()
    df['year_month'] = df['transactionmonth'].dt.to_period('M')

    monthly = df.groupby('year_month').agg({
        'totalpremium': 'sum',
        'totalclaims': 'sum',
        'loss_ratio': 'mean'
    }).reset_index()
    monthly['year_month_str'] = monthly['year_month'].astype(str)

    # Claim frequency = count of policies with claims > 0 / total policies
    claim_freq = df.groupby('year_month').apply(
        lambda x: (x['totalclaims'] > 0).sum() / len(x) * 100
    ).reset_index(name='claim_frequency_pct')

    # Average claim severity (only for policies with claims > 0)
    claim_sev = df[df['totalclaims'] > 0].groupby('year_month')['totalclaims'].mean().reset_index(name='avg_severity')

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Monthly Premium vs Claims
    x_pos = range(len(monthly))
    axes[0,0].plot(x_pos, monthly['totalpremium'], marker='o', label='Premium', color='green')
    axes[0,0].plot(x_pos, monthly['totalclaims'], marker='s', label='Claims', color='red')
    axes[0,0].set_xticks(x_pos[::2])
    axes[0,0].set_xticklabels(monthly['year_month_str'].iloc[::2], rotation=45)
    axes[0,0].set_title('Monthly Premium vs Claims', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].set_ylabel('Amount (R)')

    # 2. Loss Ratio over time
    axes[0,1].plot(x_pos, monthly['loss_ratio'], marker='o', color='purple')
    axes[0,1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Breakeven')
    axes[0,1].set_xticks(x_pos[::2])
    axes[0,1].set_xticklabels(monthly['year_month_str'].iloc[::2], rotation=45)
    axes[0,1].set_title('Average Loss Ratio Over Time', fontweight='bold')
    axes[0,1].set_ylabel('Loss Ratio')
    axes[0,1].legend()

    # 3. Claim Frequency
    axes[1,0].plot(range(len(claim_freq)), claim_freq['claim_frequency_pct'], 
                   marker='o', color='orange')
    axes[1,0].set_xticks(range(0, len(claim_freq), 2))
    axes[1,0].set_xticklabels(claim_freq['year_month'].astype(str).iloc[::2], rotation=45)
    axes[1,0].set_title('Claim Frequency (% of Policies)', fontweight='bold')
    axes[1,0].set_ylabel('% with Claims')

    # 4. Average Claim Severity
    axes[1,1].plot(range(len(claim_sev)), claim_sev['avg_severity'], 
                   marker='o', color='darkred')
    axes[1,1].set_xticks(range(0, len(claim_sev), 2))
    axes[1,1].set_xticklabels(claim_sev['year_month'].astype(str).iloc[::2], rotation=45)
    axes[1,1].set_title('Average Claim Severity', fontweight='bold')
    axes[1,1].set_ylabel('Avg Claim Amount (R)')

    fig.suptitle('Temporal Trends (Feb 2014 - Aug 2015)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_vehicle_risk_profile(df, top_n=15, figsize=(14, 10)):
    """
    Analyze vehicle makes/models with highest and lowest claim amounts.

    Returns two plots:
    1. Top N makes by average claim amount (only makes with >50 policies)
    2. Bottom N makes by average claim amount

    Why: Identifies which vehicles are "money pits" vs "safe bets".
    """
    make_stats = df.groupby('make').agg({
        'totalclaims': ['mean', 'count'],
        'loss_ratio': 'mean'
    }).reset_index()

    # Flatten column names
    make_stats.columns = ['make', 'avg_claims', 'policy_count', 'avg_loss_ratio']

    # Filter for reliability
    make_stats = make_stats[make_stats['policy_count'] >= 50]

    top_risk = make_stats.nlargest(top_n, 'avg_claims')
    low_risk = make_stats.nsmallest(top_n, 'avg_claims')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # High risk
    sns.barplot(data=top_risk, x='avg_claims', y='make', ax=axes[0], palette='Reds_d')
    axes[0].set_title(f'Top {top_n} Highest-Risk Vehicle Makes', fontweight='bold')
    axes[0].set_xlabel('Average Claim Amount (R)')

    # Low risk
    sns.barplot(data=low_risk, x='avg_claims', y='make', ax=axes[1], palette='Greens_d')
    axes[1].set_title(f'Top {top_n} Lowest-Risk Vehicle Makes', fontweight='bold')
    axes[1].set_xlabel('Average Claim Amount (R)')

    fig.suptitle('Vehicle Risk Profile Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_creative_insight_1(df, figsize=(12, 8)):
    """
    CREATIVE VISUALIZATION 1: Risk-Return Bubble Chart
    X-axis: Average Premium, Y-axis: Average Loss Ratio
    Bubble size: Number of policies
    Color: Province

    This reveals provinces that are both profitable (low loss ratio) 
    and high-volume (big bubbles) — ideal marketing targets.
    """
    prov_stats = df.groupby('province').agg({
        'totalpremium': 'mean',
        'loss_ratio': 'mean',
        'totalclaims': 'sum'
    }).reset_index()
    prov_stats['policy_count'] = df.groupby('province').size().values

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        prov_stats['totalpremium'], 
        prov_stats['loss_ratio'],
        s=prov_stats['policy_count'] / 10,  # Scale bubble size
        c=range(len(prov_stats)),
        cmap='tab10',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # Add province labels
    for i, row in prov_stats.iterrows():
        ax.annotate(row['province'], 
                   (row['totalpremium'], row['loss_ratio']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Target LR (0.5)')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Breakeven (1.0)')
    ax.set_xlabel('Average Premium per Policy (R)', fontsize=12)
    ax.set_ylabel('Average Loss Ratio', fontsize=12)
    ax.set_title('Province Risk-Return Profile\n(Bubble size = Portfolio size)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(prov_stats['loss_ratio']) * 1.2)

    plt.tight_layout()
    return fig


def plot_creative_insight_2(df, figsize=(14, 8)):
    """
    CREATIVE VISUALIZATION 2: Gender & Vehicle Type Heatmap
    Shows average Loss Ratio by Gender and Vehicle Type.

    Reveals intersectional risk patterns (e.g., are male SUV drivers 
    significantly riskier than female sedan drivers?).
    """
    pivot = df.pivot_table(
        values='loss_ratio', 
        index='vehicletype', 
        columns='gender', 
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0.5, ax=ax, linewidths=0.5)
    ax.set_title('Loss Ratio Heatmap: Vehicle Type vs Gender', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Vehicle Type', fontsize=12)

    plt.tight_layout()
    return fig


def plot_creative_insight_3(df, figsize=(14, 8)):
    """
    CREATIVE VISUALIZATION 3: Profitability Waterfall by Cover Category
    Shows how each cover category contributes to total margin.

    Positive = profitable category, Negative = loss-making category.
    This directly informs product strategy.
    """
    cover_margin = df.groupby('covercategory')['margin'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if x > 0 else 'red' for x in cover_margin.values]
    bars = ax.bar(range(len(cover_margin)), cover_margin.values, color=colors, alpha=0.8)

    ax.set_xticks(range(len(cover_margin)))
    ax.set_xticklabels(cover_margin.index, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Total Margin (R)', fontsize=12)
    ax.set_title('Profitability by Cover Category\n(Green = Profitable, Red = Loss-making)', 
                 fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, cover_margin.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'R{val:,.0f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    plt.tight_layout()
    return fig


def print_missing_value_report(df):
    """
    Generate a formatted report of missing values by column.

    Why: Missing data handling strategy must be documented for audit.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    report = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values,
        'dtype': df.dtypes.values
    })

    report = report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    print("\n=== MISSING VALUE REPORT ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns with missing values: {len(report)}")
    print("\n")
    print(report.to_string(index=False))

    return report
