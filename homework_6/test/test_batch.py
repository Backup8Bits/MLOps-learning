
from datetime import datetime

import pandas as pd

import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    categorical = ['PULocationID', 'DOLocationID']
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    data_compare = [
        ('-1', '-1', dt(1, 1), dt(1, 10)),
        ('1', '1', dt(1, 2), dt(1, 10)),
    ]

    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
    ]
    df_expected = pd.DataFrame(data_compare, columns=columns)
    print("*" * 80)
    print(df_expected)
    print("*" * 80)
    df_expected['duration'] = df_expected.tpep_dropoff_datetime - df_expected.tpep_pickup_datetime
    df_expected['duration'] = df_expected.duration.dt.total_seconds() / 60
    df = pd.DataFrame(data, columns=columns)

    df = batch.prepare_data(df, categorical)

    print(df)

    assert df.shape[0] == 2
    assert df.shape[1] == 5
    assert df_expected.equals(df)
