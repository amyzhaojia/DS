import copy
import os
import numpy as np
import pandas as pd
import sys
import unittest
import processing

class Testprocessing(unittest.TestCase):

    rdm_datalist = [3,8,6,4,59,2,7,12,79,32]
    rdm_datetime = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2021-03-01', end='2021-03-10'))]
    rdm_dataframe = pd.DataFrame({'col_a':rdm_datalist, 'date_time':rdm_datetime})

    # def generate_test_data(self, with_none=False, all_none=False):
    #     if all_none:
    #         data = []
    #         return data

    #     data = list(np.random.randint(0, 10, size=[3, 5]))
    #     if with_none:
    #         data.append([])
    #     return data

    def test_rolling_spearman(self):
        # data = self.generate_test_data(with_none=True)
        result = processing.rolling_spearman(self.rdm_dataframe['col_a'].values, self.rdm_dataframe['col_a'].values, 2)
        self.assertEqual(list(np.round(result))[1::], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # def test_decision_tree(self):
    #     # data = self.generate_test_data(all_none=True)
    #     result = processing.decision_tree(X, y, feature_names, target_name)
    #     self.assertTrue(result)

    # def test_forecasting(self):
    #     # data = self.generate_test_data(all_none=True)
    #     result = processing.forecasting(df_diff, num_data=7*24*60, interval_width=0.95)
    #     self.assertTrue(result)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Testprocessing)
    unittest.TextTestRunner(verbosity=2).run(suite)