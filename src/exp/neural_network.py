from sklearn.neural_network import MLPRegressor
from exp.data_handler import read_dataset, write_output
import pandas as pd

train_x, train_y, test_x, sk_ids = read_dataset()

nn = MLPRegressor()
nn.fit(train_x, train_y)

prediction: pd.DataFrame = list(nn.predict(test_x))

output = list(zip(sk_ids, prediction))
write_output(output)