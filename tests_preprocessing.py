import copy
import os
import numpy as np
import pandas as pd
import sys
import unittest
import preprocessing


class Testpreprocessing(unittest.TestCase):

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

    def test_multiple_sigma(self):
        # data = self.generate_test_data(with_none=True)
        result = preprocessing.multiple_sigma(self.rdm_dataframe,'col_a',2)
        self.assertEqual(result.values, [79])

    def test_box_quartile(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.box_quartile(self.rdm_dataframe['col_a'],self.rdm_dataframe['col_a'])
        self.assertEqual(result.values, [79])

    def test_poly_fit_replace(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.poly_fit_replace(self.rdm_dataframe['col_a'])
        self.assertEqual(result[1], [3, 8, 6, 4, 59.00000000000696, 2, 7, 12, 78.99999999983075, 32])

    def test_replace_outlier(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.replace_outlier(self.rdm_dataframe,'col_a',2)
        self.assertEqual(list(result['col_a'].values), [3.0,8.0,6.0,4.0,59.0,2.0,7.0,12.0,21.2,32.0])
    
    def data_reconstruction(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.data_reconstruction(self.rdm_dataframe, data_datetime_name='date_time',data_col_name='cal_a',number=3)
        self.assertEqual(result['col_a'], [3,3,3,8,8,8,6,6,6,4,4,4,59,59,59,2,2,2,7,7,7,12,12,12,79,79,79,32,32,32])
    
    def test_data_time_process(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.data_time_process(self.rdm_dataframe, datetime_name='date_time')
        self.assertEqual(list(result['datetime'].values), 
        list(pd.to_datetime(['2021-02-28 16:00:00','2021-03-01 16:00:00','2021-03-02 16:00:00','2021-03-03 16:00:00',
        '2021-03-04 16:00:00','2021-03-05 16:00:00','2021-03-06 16:00:00','2021-03-07 16:00:00',
        '2021-03-08 16:00:00','2021-03-09 16:00:00'])))
    
    def test_time_approximation(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.time_approximation(self.rdm_datetime[0])
        self.assertEqual(result, pd.to_datetime('2021-03-01 00:00:00'))
    
    def test_data_diff_process(self):
        # data = self.generate_test_data(all_none=True)
        result = preprocessing.data_diff_process(self.rdm_dataframe, 300, datetime_name='date_time')
        self.assertEqual(list(result['col_a'].values),[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Testpreprocessing)
    unittest.TextTestRunner(verbosity=2).run(suite)
