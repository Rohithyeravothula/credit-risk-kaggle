from typing import List, Tuple

import pandas as pd

# train_filename = "../../data/v1/sample_application_train.csv"
train_filename = "../../data/v1/application_train.csv"
test_filename = "../../data/v1/application_test.csv"

y_column_name = "TARGET"
sk_id_column_name = "SK_ID_CURR"

feature_columns = {'EXT_SOURCE_2', ',EXT_SOURCE_3', ',DAYS_BIRTH', ',EXT_SOURCE_1',
 ',REGION_RATING_CLIENT_W_CITY', ',REGION_RATING_CLIENT',
 ',DAYS_LAST_PHONE_CHANGE', ',NAME_EDUCATION_TYPEcat', ',CODE_GENDERcat',
 ',DAYS_ID_PUBLISH', ',REG_CITY_NOT_WORK_CITY', ',FLOORSMAX_AVG',
 ',FLOORSMAX_MEDI', ',FLOORSMAX_MODE', ',NAME_INCOME_TYPEcat',
 ',FLAG_EMP_PHONE', ',DAYS_EMPLOYED', ',REG_CITY_NOT_LIVE_CITY',
 ',FLAG_DOCUMENT_3', ',DAYS_REGISTRATION', ',TOTALAREA_MODE',
 ',YEARS_BEGINEXPLUATATION_MEDI', ',YEARS_BEGINEXPLUATATION_AVG',
 ',OCCUPATION_TYPEcat', ',YEARS_BEGINEXPLUATATION_MODE', ',LIVINGAREA_AVG',
 ',LIVINGAREA_MEDI', ',APARTMENTS_AVG', ',APARTMENTS_MEDI', ',AMT_GOODS_PRICE',
 ',EMERGENCYSTATE_MODEcat', ',LIVINGAREA_MODE', ',APARTMENTS_MODE',
 ',ENTRANCES_AVG', ',ENTRANCES_MEDI', ',REGION_POPULATION_RELATIVE',
 ',ENTRANCES_MODE', ',HOUSETYPE_MODEcat', ',ELEVATORS_AVG', ',ELEVATORS_MEDI',
 ',WALLSMATERIAL_MODEcat', ',NAME_HOUSING_TYPEcat', ',ELEVATORS_MODE',
 ',FLOORSMIN_AVG', ',FLOORSMIN_MEDI', ',FLOORSMIN_MODE', ',BASEMENTAREA_AVG',
 ',BASEMENTAREA_MEDI', ',YEARS_BUILD_MEDI', ',YEARS_BUILD_AVG',
 ',YEARS_BUILD_MODE', ',LIVE_CITY_NOT_WORK_CITY', ',DEF_30_CNT_SOCIAL_CIRCLE',
 ',BASEMENTAREA_MODE', ',DEF_60_CNT_SOCIAL_CIRCLE', ',NAME_CONTRACT_TYPEcat',
 ',ORGANIZATION_TYPEcat', ',AMT_CREDIT', ',LIVINGAPARTMENTS_AVG',
 ',LIVINGAPARTMENTS_MEDI', ',FONDKAPREMONT_MODEcat', ',LIVINGAPARTMENTS_MODE',
 ',FLAG_DOCUMENT_6', ',FLAG_WORK_PHONE'}

def convert_to_numeric(data: pd.DataFrame) -> pd.DataFrame:
    # not the best way to handle null values
    x_raw = data.fillna(value=0, axis=1)

    object_x = x_raw.select_dtypes(include=["object"]).copy()

    for col in object_x.columns:
        # brute force and generic way to handle categorical columns
        object_x[col + "cat"] = object_x[col].astype("category").cat.codes

    object_cat_x = object_x[
        [col for col in object_x.columns if col.endswith("cat")]]

    numeric_x = x_raw.select_dtypes(include=["float", "int"]).copy()

    return pd.concat([numeric_x, object_cat_x], axis=1)


def write_output(data: List[Tuple[int, int]]):
    header = "SK_ID_CURR,TARGET\n"
    str_repr = "\n".join(["{},{}".format(sk_id, target)
                          for sk_id, target in data])
    with open("../../data/output.txt", 'w') as fp:
        fp.write(header)
        fp.write(str_repr)


def read_dataset():
    """
    eliminates few columns, refer to feature_columns
    :return:
    """
    train_data = convert_to_numeric(pd.read_csv(train_filename))
    excluded_columns = {y_column_name, sk_id_column_name}
    train_y = train_data[y_column_name]
    train_x = train_data[
        [col for col in train_data.columns if col in feature_columns]]

    test_data = convert_to_numeric(pd.read_csv(test_filename))

    sk_ids = test_data[sk_id_column_name].values.tolist()
    test_x = test_data[[col for col in test_data.columns if col in feature_columns]]

    return train_x, train_y, test_x, sk_ids