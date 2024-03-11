import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import statsmodels.api as sm
def calculate_mets(cal_df, user_weights=None):
    if not user_weights or len(user_weights) == 0:
        print('no user weights provided, using default')
        user_weights = dict(zip(cal_df.user_id.unique(), np.ones(cal_df.user_id.nunique()) * 70))
    mets_df = cal_df.copy()
    mets_df['value'] = mets_df['value'] * 4.186

    mets_df['value'] = mets_df.apply(lambda x: x['value'] / (user_weights[x['user_id']]), axis=1)

    grouped = mets_df.groupby('user_id')

    calibrated_df = pd.DataFrame()
    for name, group in grouped:
        group['datetime'] = pd.to_datetime(group.index)
        # normalize user's mets mode to 1.0
        baseline = 1.00 / group['value'].mean()
        group['value'] = group['value'] * baseline

        group['days_since_start'] = (group.datetime - group.datetime.iloc[0]).dt.total_seconds() / (24 * 3600)
        group['value'] = np.where(group['days_since_start'].diff().shift(-1) > 0.5, None, group['value'])

        calibrated_df = pd.concat([calibrated_df, group])

    # replace index
    calibrated_df.reset_index(drop=True, inplace=True)
    return calibrated_df
    # return pd.DataFrame(columns=['user_id', 'timestamp', 'value'])

def clean_weight_and_plot(weight_df, start_date, end_date):
    weight_df['date'] = pd.to_datetime(weight_df['date'])
    weight_df = weight_df[['date', 'weight_kg']]
    weight_df = weight_df[(weight_df['date'].dt.date >= start_date) & (weight_df['date'].dt.date <= end_date)]

    # filter outliers using quantile
    Q1 = weight_df['weight_kg'].quantile(0.25)
    Q3 = weight_df['weight_kg'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    weight_df = weight_df[(weight_df['weight_kg'] >= lower_bound) & (weight_df['weight_kg'] <= upper_bound)]

    weight_df['is_interpolate'] = False

    weight_df.reset_index()
    # date_list = weight_df['date'].apply(lambda x: x.strftime('%Y-%m-%d')).values

    # # fill records for missing days
    date_range = pd.date_range(start=start_date, end=end_date)
    # for date in date_range:
    #     if date.strftime('%Y-%m-%d') not in date_list:
    #         weight_df = pd.concat([weight_df,pd.DataFrame([{'date': pd.to_datetime(date), 'weight_kg': np.nan, 'is_interpolate':True}])])

    # Create a DataFrame containing all dates in date_range
    date_range_df = pd.DataFrame(date_range, columns=['date'])

    # Convert the date in weight_df to string format
    weight_df['date_str'] = weight_df['date'].dt.strftime('%Y-%m-%d')

    # Find dates that do not exist in weight_df
    missing_dates = date_range_df[~date_range_df['date'].dt.strftime('%Y-%m-%d').isin(weight_df['date_str'])]

    # Create records for these missing dates
    missing_records = missing_dates.assign(weight_kg=np.nan, is_interpolate=True)

    # Merge these records with original weight_df
    weight_df = pd.concat([weight_df, missing_records], ignore_index=True)

    # Optionally remove the auxiliary 'date_str' column
    weight_df.drop(columns=['date_str'], inplace=True)

    weight_df['date'] = pd.to_datetime(weight_df['date'])
    weight_df = weight_df.sort_values(by='date')
    weight_df = weight_df.reset_index(drop=True)

    # interpolate
    weight_df['weight_kg'].interpolate(method='linear', limit_direction='both', inplace=True)

    weight_df['date'] = pd.to_datetime(weight_df['date'])
    outliers = weight_df[weight_df['is_interpolate']]

    fig = go.Figure()
    # weight_df = weight_df.drop(weight_df.index)
    # outliers = outliers.drop(outliers.index)
    fig.add_trace(go.Scatter(x=weight_df['date'], y=weight_df['weight_kg'], mode='lines', name='Weight (kg)',
                             marker=dict(size=3)))

    fig.add_trace(go.Scatter(x=outliers['date'], y=outliers['weight_kg'], mode='markers', name='Interpolation Value',
                             marker=dict(color='red', size=3, symbol='hexagon')))

    fig.update_layout(
        title='Weight vs. Date',
        xaxis_title='Date',
        yaxis_title='Weight (kg)',
        legend_title="Legend",
        template="plotly_white",
        width=1200,
        height=400
    )

    # fig.show()
    # weight_html = fig.to_html(include_plotlyjs='cdn')
    return weight_df, fig

def generate_activity_level(merged_df):
    """
        This function generates a Plotly figure illustrating the patient's activity level over time. The figure includes both METS values and the smoothed activity level, using different y-axes.

        Parameters:
            merged_df (pandas.DataFrame): A DataFrame containing the following columns:
                - 'date' (datetime or string): The date of the activity.
                - 'time' (datetime or string): The specific time of the activity.
                - 'mets' (float): The Metabolic Equivalent of Task (MET) values representing the intensity of activities.
                - 'activity_duration' (float): The duration of the activity sessions, with longer durations represented by darker colors.
                - 'smooth_activity' (float): The smoothed activity level, calculated from the raw activity data to show trends over time.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure with two main features:
                - A scatter plot on the left y-axis representing individual activity sessions. The dots represent different METS values and are colored based on the activity duration, with darker colors indicating longer sessions.
                - A line plot on the right y-axis representing the smoothed activity level over time, which shows the overall trend in the patient's activity levels.

        Example:
            >>> activity_fig = create_activity_level_plot(merged_df)
            >>> activity_fig.show()

        Note:
            Ensure that 'merged_df' contains all required columns before passing it to this function. Missing data may result in errors or an incorrect plot output.
        """
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, float('inf')]

    def binning(group):
        group['mets_bin'] = pd.cut(group['mets'], bins=bins, right=False)
        return group

    merged_df = merged_df.groupby('date', group_keys=False).apply(binning)

    result_df = merged_df.groupby(['date', 'mets_bin']).agg(avg_mets=('mets', 'mean'),
                                                            record_num=('mets', 'count')).reset_index()

    result_df = result_df.dropna(subset=['avg_mets'])
    result_df['record_num'] = result_df.groupby('date', group_keys=False)['record_num'].apply(lambda x: x / x.sum())
    # print(result_df)

    # calculate ci
    weighted_avg = result_df.groupby('date').apply(lambda x: np.average(x['avg_mets'], weights=x['record_num']))

    # Calculate weighted standard error
    def weighted_std(x):
        average = np.average(x['avg_mets'], weights=x['record_num'])
        variance = np.average((x['avg_mets'] - average) ** 2, weights=x['record_num'])
        return np.sqrt(variance)

    weighted_se = result_df.groupby('date').apply(weighted_std)
    # Calculate the upper and lower bounds of the 95% confidence interval
    ci_upper = weighted_avg + 1.96 * weighted_se
    ci_lower = weighted_avg - 1.96 * weighted_se

    # print(ci_upper)

    def weighted_avg(group):
        weights = group['record_num']
        values = group['avg_mets']
        return np.average(values, weights=weights)

    weighted_avg_per_date = result_df.groupby('date').apply(weighted_avg).reset_index(name='weighted_avg_mets')
    # print(weighted_avg_per_date)

    weighted_avg_per_date['date'] = pd.to_datetime(weighted_avg_per_date['date'])
    weighted_avg_per_date['date_num'] = (weighted_avg_per_date['date'] - weighted_avg_per_date['date'].min()).dt.days

    # LOWESS generate smooth line
    lowess = sm.nonparametric.lowess
    z = lowess(weighted_avg_per_date['weighted_avg_mets'], weighted_avg_per_date['date_num'], frac=0.35)  # frac是平滑参数
    color_scale = np.interp(result_df['record_num'], (result_df['record_num'].min(), result_df['record_num'].max()),
                            [0, 1])
    fig = go.Figure()
    activity_level_point = result_df

    # points
    fig.add_trace(go.Scatter(
        x=activity_level_point['date'], y=activity_level_point['avg_mets'], mode='markers',
        marker=dict(color=color_scale, colorscale='Greens', size=10),
        name='activity point'
    ))

    # second Y axis
    fig.update_layout(
        yaxis2=dict(
            title="regressive level",
            anchor="x",
            overlaying="y",
            side="right",
        )
    )

    # line
    fig.add_trace(go.Scatter(
        x=weighted_avg_per_date['date'], y=z[:, 1], mode='lines',
        name='activity regressive line', yaxis="y2"
    ))

    # Add confidence interval
    # fig.add_trace(go.Scatter(x=weighted_avg_per_date['date'], y=ci_upper, mode='lines', name='ci_upper', line=dict(width=0),yaxis="y2"))
    # fig.add_trace(go.Scatter(x=weighted_avg_per_date['date'], y=ci_lower, mode='lines', name='ci_lower', line=dict(width=0), fill='tonexty',yaxis="y2"))

    fig.update_layout(title='regression analysis of activity level', xaxis_title='time', yaxis_title='activity level',
                      width=1200, height=400)
    # fig.show()
    activity_level_html = fig.to_html(include_plotlyjs='cdn')
    return fig


def categorize_mets(merged_df):
    """
    Categorize METs values into activity levels and calculate time spent in each category.

    Input:
        merged_df (DataFrame): A DataFrame with columns including 'mets', 'date', and 'time'.

    Process:
        Categorizes METs values into activity levels (sedentary, lightly_active, fairly_active, very_active),
        then groups data by date and activity level to calculate time spent in each category.

    Output:
        Returns a DataFrame mets_df with dates, categorized activity levels,
        and the total time spent in each activity level, including 'total_active'
        and 'non-sedentary' summaries.
    """
    def divide_mets(mets):
        if mets < 1.5:
            return 'sedentary'
        elif 1.5 <= mets < 3.0:
            return 'lightly_active'
        elif 3.0 <= mets < 6.0:
            return 'fairly_active'
        else:
            return 'very_active'

    merged_df['mets_category'] = merged_df['mets'].apply(divide_mets)

    # categorized by date and level
    grouped = merged_df.groupby(['date', 'mets_category'])['time'].count() / 60
    mets_df = grouped.unstack().reset_index().fillna(0)
    # mets_df.columns.name = None

    if 'sedentary' not in mets_df.columns:
        mets_df['sedentary'] = 0
    if 'lightly_active' not in mets_df.columns:
        mets_df['lightly_active'] = 0
    if 'fairly_active' not in mets_df.columns:
        mets_df['fairly_active'] = 0
    if 'very_active' not in mets_df.columns:
        mets_df['very_active'] = 0

    mets_df['total_active'] = mets_df[['sedentary', 'lightly_active', 'fairly_active', 'very_active']].sum(axis=1)
    mets_df['non-sedentary'] = mets_df[['lightly_active', 'fairly_active', 'very_active']].sum(axis=1)

    return mets_df


def generate_activity_hours(mets_df, ae_date=None):
    """
    `generate_activity_hours`

    ## Input
    - `mets_df`: A DataFrame with columns including `date` and various activity levels like `sedentary`, `lightly_active`, `fairly_active`, `very_active`, and `total_active`.
        See also return from function 'categorize_mets'
    - `ae_date`: Optional; a specific date to highlight on the plots as 'AE' (Adverse Event). Default is None.

    ## Process
    Creates a figure with six subplots, each representing time spent in different activity levels across dates. It visualizes data for 'Non-Sedentary Hours', 'Sedentary Hours', 'Lightly Active Hours', 'Fairly Active Hours', 'Very Active Hours', and 'Total Active Hours'. It uses different colors for different activity types and adds a vertical line and annotation on all subplots if `ae_date` is provided.

    ## Output
    Returns a Plotly `fig` object that contains the subplot figure with the activity data visualized across different levels. This can be used directly with `fig.show()` to display the figure or saved as an HTML or static image file.
    """
    titles = ['Non-Sedentary Hours', 'Sedentary Hours', 'Lightly Active Hours', 'Fairly Active Hours',
              'Very Active Hours', 'Total Active Hours']
    fig = make_subplots(rows=6, cols=1, subplot_titles=titles)

    activity_types = ['non-sedentary', 'sedentary', 'lightly_active', 'fairly_active', 'very_active', 'total_active']
    colors = ['#EEB422', '#99CCFF', '#99CC99', '#FF9933', '#FF99CC', '#000000']
    for i, activity in enumerate(activity_types):
        fig.add_trace(
            go.Scatter(x=mets_df['date'], y=mets_df[activity], mode='markers+lines', name=activity,
                       marker=dict(color=colors[i]), line=dict(width=2)),
            row=i + 1, col=1
        )
        # fig.add_annotation(xref='paper', yref='paper', x=0.5, y=1-(i/6)-0.03, showarrow=False, font=dict(size=12), row=i+1, col=1)

    if ae_date is not None:
        for i in range(1, 7):
            fig.add_vline(x=pd.Timestamp(ae_date), line_color="red", line_dash="dash", row=i, col=1)
            fig.add_annotation(x=pd.Timestamp(ae_date), y=15, text=f'AE: {ae_date}', showarrow=False,
                               font=dict(color='red'), row=i, col=1)
    fig.add_hline(y=8, line_color="blue", line_dash="dash", row=6, col=1)

    fig.update_layout(height=1000, width=900, title_text="Activity Analysis", showlegend=False)
    # fig.update_annotations(dict(xref="x", yref="y"))


    return fig

