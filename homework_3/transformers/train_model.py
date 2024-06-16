from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs):
    columns = ["PULocationID", "DOLocationID"]
    target = "duration"
    df_one_hot = data[columns]
    df_one_hot = df_one_hot.astype(str)
    dv_df = df_one_hot.to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(dv_df)
    y_train = data[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return dv, lr