import pickle
import pandas as pd
import numpy as np
import click

# year = 2023
# month = 3


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


@click.command()
@click.option("-y", '--year', required=True, type=int)
@click.option("-m", '--month', required=True, type=int)
def calculate(year, month):
    print(f"year: {year}, month: {month}")
    file_name = "https://d37ci6vzurychx.cloudfront.net/" +\
        f"trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"{year}-{month:02d}-processed.parquet"

    df = read_data(file_name)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    std_dev = np.std(y_pred)
    print("standard deviation", std_dev)

    df["ride_id"] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['prediction'] = y_pred
    df_result = df[['ride_id', 'prediction']].copy()

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    mean_pred = np.mean(y_pred)
    print("mean prediction", mean_pred)


if __name__ == '__main__':
    calculate()
