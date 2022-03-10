import pandas as pd
import numpy as np
import scipy as sp
from itertools import combinations
import math

class shap():
    def __init__(self):
        _ = 1

    def shap_values(self, x, y):

        if isinstance(x, pd.DataFrame):
            self.x = x.values
        elif isinstance(x, np.ndarray):
            self.x = x
        else:
            self.x = np.array(x)

        if isinstance(y, pd.DataFrame):
            self.y = y.values.reshape(-1)
        elif isinstance(x, np.ndarray):
            self.y = y.reshape(-1)
        else:
            self.y = np.array(y).reshape(-1)

        self.column_size = self.x.shape[1]

        shapley_values = np.array([])
        # caluculate shapley value for each column
        for c in range(self.column_size):
            # subset of columns which excludes target columns
            target_rows_x = self.x[self.x[:, c] == 0,:]
            target_rows_y = self.y[self.x[:, c] == 0]

            shapley_value = 0
            # calculate marginal contribution and its coefficient
            for target_row_x, target_row_y in zip(target_rows_x, target_rows_y):
                # find subset include the target subset and x_i
                row_cp = target_row_x.copy()
                np.put(row_cp, [c], 1)
                y_cp = self.y[(self.x == np.array(row_cp)).all(axis=1)]

                if y_cp is not None and len(y_cp) > 0:
                    comb = (
                        math.factorial(target_row_x.sum()) *
                        math.factorial(self.column_size - target_row_x.sum() - 1)
                    ) / math.factorial(self.column_size)
                    marginal_cont = y_cp[0] - target_row_y
                    shapley_value += comb * marginal_cont

            # marginal contribution for empty set
            row_empty = np.zeros(self.column_size)
            np.put(row_empty, [c], 1)
            y_cp = self.y[(self.x == np.array(row_empty)).all(axis=1)]
            if y_cp is not None and len(y_cp) > 0:
                comb = math.factorial(self.column_size - 1) / math.factorial(self.column_size)
                marginal_cont = y_cp[0]
                shapley_value += comb * marginal_cont

            shapley_values = np.append(shapley_values, shapley_value)

        return shapley_values


    def simple_shap_values(self, x, y):

        if isinstance(x, pd.DataFrame):
            self.x = x.values
        elif isinstance(x, np.ndarray):
            self.x = x
        else:
            self.x = np.array(x)

        if isinstance(y, pd.DataFrame):
            self.y = y.values.reshape(-1)
        elif isinstance(x, np.ndarray):
            self.y = y.reshape(-1)
        else:
            self.y = np.array(y).reshape(-1)

        self.column_size = self.x.shape[1]

        shapley_values = np.array([])
        # caluculate shapley value for each column
        for c in range(self.column_size):
            # subset of columns which excludes target columns
            target_rows_x = self.x[self.x[:, c] == 0,:]
            # target_rows_y = self.y[self.x[:, c] == 0]

            shapley_value = 0
            # calculate marginal contribution and its coefficient
            for target_row_x in target_rows_x:
                # find subset include the target subset and x_i
                row_cp = target_row_x.copy()
                np.put(row_cp, [c], 1)
                y_cp = self.y[(self.x == np.array(row_cp)).all(axis=1)]

                if y_cp is not None and len(y_cp) > 0:
                    comb = 1 / (target_row_x.sum() + 1)
                    marginal_cont = y_cp[0]
                    shapley_value += comb * marginal_cont

            # marginal contribution for empty set
            row_empty = np.zeros(self.column_size)
            np.put(row_empty, [c], 1)
            y_cp = self.y[(self.x == np.array(row_empty)).all(axis=1)]
            if y_cp is not None and len(y_cp) > 0:
                shapley_value += y_cp[0]

            shapley_values = np.append(shapley_values, shapley_value)

        return shapley_values
