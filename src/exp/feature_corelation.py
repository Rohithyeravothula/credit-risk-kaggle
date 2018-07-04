from exp.data_handler import read_dataset, convert_to_numeric, train_filename
import pandas as pd
from scipy.stats import pearsonr


train_data = convert_to_numeric(pd.read_csv(train_filename))

column_names = train_data.columns[2:]
target = "TARGET"


corelations = {}
for f in column_names:
    data_temp = train_data[[f, target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    corelations[key] = pearsonr(x1, x2)[0]


data_corelation = pd.DataFrame(corelations, index=["Value"]).T

