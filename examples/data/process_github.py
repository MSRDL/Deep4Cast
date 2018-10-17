import pandas as pd
import argparse

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def prepare_data(data_path):
    """Returns dataframe with features."""

    # Get data
    df = pd.read_csv(data_path)

    # Remove NaNs
    df = df.dropna()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df.date)

    # Create and age variable
    df['age'] = df.index.astype('int')

    # Create a day of week field
    df['day'] = df.date.dt.dayofweek

    # Create a month of year field
    df['month'] = df.date.dt.month

    # Create a boolean for US federal holidays
    holidays = calendar().holidays(start=df.date.min(), end=df.date.max())
    df['holiday'] = df['date'].isin(holidays).apply(int)

    # Rearrange columns
    df = df[
        [
            'date',
            'count',
            'age',
            'month',
            'day',
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

    # Reset index
    df = df.reset_index(drop=True)

    return df


def run():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    # Get the data
    df = prepare_data(args.data_path)

    # Store data
    df.to_pickle(args.output_path)


if __name__ == '__main__':
    run()
