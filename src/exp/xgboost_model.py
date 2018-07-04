from xgboost import XGBRegressor
import pandas as pd
from exp.data_handler import write_output, read_dataset

train_x, train_y, test_x, sk_ids = read_dataset()

boost_model = XGBRegressor()

boost_model.fit(train_x, train_y)

prediction: pd.DataFrame = [p if p >= 0 else 0 for p in
                            list(boost_model.predict(test_x))]

output = list(zip(sk_ids, prediction))
write_output(output)