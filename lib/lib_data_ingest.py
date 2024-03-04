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