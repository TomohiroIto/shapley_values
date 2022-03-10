import unittest
from shap_module import shap

class ShapModuleTest(unittest.TestCase):
    def test_shap_sample(self):
        # data from https://qiita.com/shiibass/items/02ccef36ed04b876e2e4
        # contributed players
        x = [
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ]

        # profit
        y = [[40, 20, 65, 0, 50, 30, 100]]

        s = shap()
        print(s.shap_values(x, y))

if __name__ == "__main__":
    unittest.main()
