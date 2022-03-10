import pandas as pd
import numpy as np
from itertools import product
import math

class shap():
    def __init__(self):
        _ = 1

    def _conv_profit_to_contribution(self, x, y):

        if isinstance(x, pd.DataFrame):
            xp = x.values
        elif isinstance(x, np.ndarray):
            xp = x
        else:
            xp = np.array(x)

        if isinstance(y, pd.DataFrame):
            yp = y.values.reshape(-1)
        elif isinstance(x, np.ndarray):
            yp = y.reshape(-1)
        else:
            yp = np.array(y).reshape(-1)

        column_size = xp.shape[1]
        subsets = np.array(list(product(*[[0, 1] for d in range(column_size)])))

        y_ret = np.array([])
        for subset in subsets:
            new_y = 0
            subsubsets = np.array(list(product(*[[0] if d == 0 else [0, 1] for d in subset])))
            for subsubset in subsubsets:
                y_found = yp[(xp == subsubset).all(axis=1)]
                if y_found is not None and len(y_found) > 0:
                    new_y += y_found[0]

            y_ret = np.append(y_ret, new_y)

        return subsets, y_ret


    def shap_values(self, x, y):
        self.x, self.y = self._conv_profit_to_contribution(x, y)
        self.column_size = self.x.shape[1]

        # caluculate shapley value for each column
        shapley_values = np.array([])
        for c in range(self.column_size):
            # subset of columns which excludes current column
            target_rows_x = np.array(list(product(*[[0] if d == c else [0, 1] for d in range(self.column_size)])))

            # calculate marginal contribution and its coefficient
            shapley_value = 0
            for target_row_x in target_rows_x:
                # find subset include the current subset and x_i
                row_cp = target_row_x.copy()
                np.put(row_cp, [c], 1)
                target_row_y = self.y[(self.x == np.array(target_row_x)).all(axis=1)]
                y_cp = self.y[(self.x == np.array(row_cp)).all(axis=1)]

                if target_row_y is None or len(target_row_y) == 0:
                    target_row_y = 0
                if y_cp is None or len(y_cp) == 0:
                    y_cp = 0

                comb = (
                    math.factorial(target_row_x.sum()) *
                    math.factorial(self.column_size - target_row_x.sum() - 1)
                ) / math.factorial(self.column_size)
                marginal_cont = y_cp[0] - target_row_y
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

        # caluculate shapley value for each column
        shapley_values = np.array([])
        for c in range(self.column_size):
            # subset of columns which excludes current column
            target_rows_x = np.array(list(product(*[[0] if d == c else [0, 1] for d in range(self.column_size)])))

            # calculate marginal contribution and its coefficient
            shapley_value = 0
            for target_row_x in target_rows_x:
                # find subset include the current subset and x_i
                row_cp = target_row_x.copy()
                np.put(row_cp, [c], 1)
                y_cp = self.y[(self.x == np.array(row_cp)).all(axis=1)]

                if y_cp is not None and len(y_cp) > 0:
                    comb = 1 / (target_row_x.sum() + 1)
                    marginal_cont = y_cp[0]
                    shapley_value += comb * marginal_cont

            shapley_values = np.append(shapley_values, shapley_value)

        return shapley_values
