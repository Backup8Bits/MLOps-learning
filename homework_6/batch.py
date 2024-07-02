#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd


# year = int(sys.argv[1])
# month = int(sys.argv[2])

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'test')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL,
        'use_ssl': False,
        # 'aws_access_key_id': AWS_ACCESS_KEY_ID,
        # 'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
    }
}


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def read_by_path(filename):
    df = pd.read_parquet(filename, storage_options=options)

    return df


def read_data(filename:str, categorical):
    df = read_by_path(filename)
    df = prepare_data(df, categorical)

    return df


def save_data(df_result, output_file):
    df_result.to_parquet(
        output_file, engine='pyarrow', index=False, storage_options=options
    )


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/' +\
        f'trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)

    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/' +\
        f'taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)

    return output_pattern.format(year=year, month=month)


# main funtion with parameters year and month
def main(year:str, month:str):
    year = int(year)
    month = int(month)
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print('predicted sum duration:', y_pred.sum())
    save_data(df_result, output_file)


if __name__ == '__main__': ## First Question
    # TEMP TESTING
    year = "2023"
    month = "1"
    main(year, month)
