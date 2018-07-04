import pandas as pd
from sklearn.linear_model import LinearRegression

from exp.data_handler import read_dataset, write_output

train_x, train_y, test_x, sk_ids = read_dataset()


lg = LinearRegression()
lg.fit(train_x, train_y)


prediction: pd.DataFrame = list(lg.predict(test_x))

output = list(zip(sk_ids, prediction))
write_output(output)


