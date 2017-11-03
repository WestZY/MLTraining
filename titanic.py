# coding: utf-8

import numpy as np
import pandas as pd

data = pd.read_csv('./titanic/train.csv')

Y = data['Survived'].copy()
print Y
