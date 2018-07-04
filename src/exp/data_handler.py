from typing import List, Tuple

import pandas as pd

# train_filename = "../../data/v1/sample_application_train.csv"
train_filename = "../../data/v1/application_train.csv"
test_filename = "../../data/v1/application_test.csv"

y_column_name = "TARGET"
sk_id_column_name = "SK_ID_CURR"

feature_columns = {'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                   'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                   'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                   'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                   'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                   'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                   'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                   'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                   'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
                   'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                   'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
                   'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                   'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                   'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                   'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                   'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
                   'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                   'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                   'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
                   'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
                   'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
                   'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
                   'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                   'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
                   'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
                   'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                   'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
                   'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
                   'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                   'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
                   'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
                   'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
                   'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                   'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
                   'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
                   'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                   'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                   'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                   'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                   'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                   'AMT_REQ_CREDIT_BUREAU_YEAR', 'FLAG_DOCUMENT_3'}

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
    train_data = convert_to_numeric(pd.read_csv(train_filename))
    excluded_columns = {y_column_name, sk_id_column_name}
    train_y = train_data[y_column_name]
    train_x = train_data[
        [col for col in train_data.columns if col in feature_columns]]

    test_data = convert_to_numeric(pd.read_csv(test_filename))

    sk_ids = test_data[sk_id_column_name].values.tolist()
    test_x = test_data[[col for col in test_data.columns if col in feature_columns]]

    return train_x, train_y, test_x, sk_ids