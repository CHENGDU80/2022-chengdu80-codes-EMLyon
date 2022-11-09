from typing import Optional, Union

import pandas as pd
import numpy as np


def load(
    data: Union[str, pd.DataFrame],
    label: Optional[Union[str, pd.DataFrame]] = None,
    on: Optional[str] = None,
    categorical_columns: list = [],
    datetime_columns: list = [],
    numeric_columns: list = [],
    **kwargs,
) -> pd.DataFrame:
    """Load data."""
    if isinstance(data, str):
        data = pd.read_csv(data, **kwargs)
    if isinstance(label, str):
        label = pd.read_csv(label, **kwargs)
    if label is not None:
        data = pd.merge(data, label, how='inner', on=on)
    # TODO: process categorical series
    for column in categorical_columns:
        pass
    # TODO: process datetime series
    for column in datetime_columns:
        data[column] = pd.to_datetime(data[column])
    # TODO: process numeric series
    for column in numeric_columns:
        pass
    return data


def nisna(series: pd.Series) -> int:
    return series.isna().sum()


def pctisna(series: pd.Series) -> float:
    return nisna(series) / len(series) * 100


def describe(data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics of data."""
    statistics = data.describe(include='all', datetime_is_numeric=True)
    extra = data.agg([nisna, pctisna, 'nunique', 'dtype'])
    return pd.concat([statistics, extra])


def get_categorical_columns(
    data: pd.DataFrame,
    feature_columns: Optional[list] = None,
    include=[object, 'category'],
    exclude: Optional[list] = None,
) -> list:
    feature = data if feature_columns is None else data[feature_columns]
    return list(feature.select_dtypes(include=include, exclude=exclude))


def get_datetime_columns(
    data: pd.DataFrame,
    feature_columns: Optional[list] = None,
    include: list = ['datetime'],
    exclude: Optional[list] = None,
) -> list:
    feature = data if feature_columns is None else data[feature_columns]
    return list(feature.select_dtypes(include=include, exclude=exclude))


def get_numeric_columns(
    data: pd.DataFrame,
    feature_columns: Optional[list] = None,
    include: list = ['number'],
    exclude: Optional[list] = None,
) -> list:
    feature = data if feature_columns is None else data[feature_columns]
    return list(feature.select_dtypes(include=include, exclude=exclude))


def select(
    data: pd.DataFrame,
    statistics: pd.DataFrame,
    feature_columns: Optional[list] = None,
    thres_pctisna: Optional[float] = None,
    thres_nunique: Optional[int] = None,
    as_pandas: bool = True,
) -> pd.DataFrame:
    """Select columns by thresholds on statistics of data."""
    feature = data if feature_columns is None else data[feature_columns]
    statistics = statistics[list(feature)].T
    result = {}
    # categorical columns
    categorical_columns = get_categorical_columns(feature)
    if categorical_columns:
        message = '{} categorical column(s): {}'.format(
            len(categorical_columns),
            str(categorical_columns)[1:-1],
        )
    else:
        message = 'no categorical column'
    result['categorical'] = [
        categorical_columns,
        len(categorical_columns), message
    ]
    # datetime columns
    datetime_columns = get_datetime_columns(feature)
    if datetime_columns:
        message = '{} datetime column(s): {}'.format(
            len(datetime_columns),
            str(datetime_columns)[1:-1],
        )
    else:
        message = 'no datetime column'
    result['datetime'] = [datetime_columns, len(datetime_columns), message]
    # numeric columns
    numeric_columns = get_numeric_columns(feature)
    if numeric_columns:
        message = '{} numeric column(s): {}'.format(
            len(numeric_columns),
            str(numeric_columns)[1:-1],
        )
    else:
        message = 'no numeric column'
    result['numeric'] = [numeric_columns, len(numeric_columns), message]
    dropped_columns = []
    # dropped columns by missing value ratio, `pctisna`
    if thres_pctisna is not None:
        droppedby_pctisna = list(
            statistics.loc[statistics['pctisna'] > thres_pctisna].index)
        if droppedby_pctisna:
            message = '{} column(s) whose missing value ratio is greater than {}%: {}'.format(
                len(droppedby_pctisna),
                thres_pctisna,
                str(droppedby_pctisna)[1:-1],
            )
        else:
            message = 'no column whose missing value ratio is greater than {}%'.format(
                thres_pctisna)
        result['droppedby_pctisna'] = [
            droppedby_pctisna,
            len(droppedby_pctisna),
            message,
        ]
        dropped_columns += droppedby_pctisna
    else:
        result['droppedby_pctisna'] = [np.nan, np.nan, np.nan]
    # dropped columns by unique value number, `nunique`
    if thres_nunique is not None:
        droppedby_nunique = list(
            statistics.loc[statistics['nunique'] < thres_nunique].index)
        if droppedby_nunique:
            message = '{} column(s) whose unique value number is smaller than {}: {}'.format(
                len(droppedby_nunique),
                thres_nunique,
                str(droppedby_nunique)[1:-1],
            )
        else:
            message = 'no column whose unique value number is smaller than {}'.format(
                thres_nunique)
        result['droppedby_nunique'] = [
            droppedby_nunique,
            len(droppedby_nunique),
            message,
        ]
        dropped_columns += droppedby_nunique
    else:
        result['droppedby_nunique'] = [np.nan, np.nan, np.nan]
    # dropped columns
    dropped_columns = list(set(dropped_columns))
    if dropped_columns:
        message = '{} dropped column(s): {}'.format(
            len(dropped_columns),
            str(dropped_columns)[1:-1],
        )
    else:
        message = 'no dropped column'
    result['dropped'] = [dropped_columns, len(dropped_columns), message]
    # feature columns
    feature = feature.drop(columns=dropped_columns)
    feature_columns = list(feature)
    if feature_columns:
        message = '{} left column(s): {}'.format(
            len(feature_columns),
            str(feature_columns)[1:-1],
        )
    else:
        message = 'no left column'
    result['left'] = [feature_columns, len(feature_columns), message]
    if as_pandas:
        return pd.DataFrame.from_dict(
            result,
            orient='index',
            columns=['columns', 'count', 'comment'],
        )
    return result
