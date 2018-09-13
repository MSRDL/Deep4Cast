import pandas as pd
import argparse


def prepare_data(data_path):
    """Returns dataframe with features."""

    # Get data
    df = pd.read_csv(data_path)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df.date)
    df = df.set_index('date').resample('B').last()

    # Drop NaNs
    df = df.dropna()

    # Moving averages
    df['200_day_ma'] = df.close_price.rolling(window=200).mean()
    df['50_day_ma'] = df.close_price.rolling(window=50).mean()

    # Create a return field
    df['return'] = df['close_price'].pct_change()
    df = df.reset_index()

    # Create a day of week field
    df['day'] = df.date.dt.dayofweek

    # Create a month of year field
    df['month'] = df.date.dt.month

    # Rearrange columns
    df = df[
        [
            'date',
            'close_price',
            '200_day_ma',
            '50_day_ma',
            'return',
            'month',
            'day'
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
