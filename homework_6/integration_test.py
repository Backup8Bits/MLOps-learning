import os
from datetime import datetime

import pandas as pd


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


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def main():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    input_file = 's3://nyc-duration/in/2023-01.parquet'

    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


if __name__ == '__main__':
    main()