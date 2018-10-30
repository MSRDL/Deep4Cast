import datetime as dt
import pandas as pd
import argparse
import os

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def prepare_data(data_path):
    """Returns dataframe with features."""

    # Get data
    df = pd.read_csv(data_path)

    # Remove NaNs
    df = df.dropna()

    # Rename columns for consistency
    df = df.rename(
        columns={'Hour': 'hour', 'Date': 'date', 'T': 'temperature'})

    # Convert date to datetime
    df['date'] = pd.to_datetime(df.date)

    # Create a time field
    df['time'] = df.date + \
        df.hour.apply(lambda x: dt.timedelta(hours=float(x)))

    # Create a day of week field
    df['day'] = df.time.dt.dayofweek

    # Create a month of year field
    df['month'] = df.time.dt.month

    # Create a boolean for US federal holidays
    holidays = calendar().holidays(start=df.date.min(), end=df.date.max())
    df['holiday'] = df['date'].isin(holidays).apply(int)

    # Rearrange columns
    df = df[
        [
            'time',
            'month',
            'day',
            'hour',
            'load',
            'temperature',
            'holiday'
        ]
    ]

    # Create monthly dummies
    tmp = pd.get_dummies(df.month)
    tmp.columns = ['month' + str(value) for value in tmp.columns]
    df = pd.concat([df, tmp], axis=1)

    # Create daily dummies
    tmp = pd.get_dummies(df.day)
    tmp.columns = ['day' + str(value) for value in tmp.columns]
    df = pd.concat([df, tmp], axis=1)

    # Create hourly dummies
    tmp = pd.get_dummies(df.hour)
    tmp.columns = ['hour' + str(value) for value in tmp.columns]
    df = pd.concat([df, tmp], axis=1)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_path',
                        default='raw/gefcom2014-e.csv')
    parser.add_argument('--output_path',
                        default='processed/GEFCom2014-e.pkl')
    args = parser.parse_args()

    # Get the data
    df = prepare_data(args.data_path)
        
    # Store data
    path = os.path.dirname(args.output_path)
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_pickle(args.output_path)


if __name__ == '__main__':
    main()
