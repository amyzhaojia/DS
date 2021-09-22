import copy
import os
import numpy as np
import pandas as pd
import sys
import unittest
import Plot_ML

class TestPlot_ML(unittest.TestCase):

    rdm_datalist = [3,8,6,4,59,2,7,12,79,32]
    rdm_datalist1 = [2,9,47,7,8,79,62,4,57,15]
    rdm_datetime = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2021-03-01', end='2021-03-10'))]
    target_name = ['ISnt_Alert', 'Is_Alert']
    rdm_label = [0,1,1,0,1,0,0,0,0,0]
    rdm_dataframe = pd.DataFrame({'col_a':rdm_datalist, 'date_time':rdm_datetime, 'col_b':rdm_datalist1})

    # def generate_test_data(self, with_none=False, all_none=False):
    #     if all_none:
    #         data = []
    #         return data

    #     data = list(np.random.randint(0, 10, size=[3, 5]))
    #     if with_none:
    #         data.append([])
    #     return data

    def test_pie_figure(self):
        # data = self.generate_test_data(with_none=True)
        result = Plot_ML.pie_figure([self.rdm_label.count(1),self.rdm_label.count(0)], self.target_name)
        self.assertTrue(result)

    def test_venn_figure(self):
        # data = self.generate_test_data(all_none=True)
        result = Plot_ML.venn_figure([self.rdm_label,self.rdm_label], self.target_name)
        self.assertTrue(result)

    def test_cor_fig(self):
        # data = self.generate_test_data(all_none=True)
        result = Plot_ML.cor_fig(self.rdm_dataframe)
        self.assertTrue(result)

    def test_plot_double_axes(self):
        # data = self.generate_test_data(all_none=True)
        result = Plot_ML.plot_double_axes(self.rdm_dataframe['col_a'],self.rdm_dataframe['col_b'])
        self.assertTrue(result)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPlot_ML)
    unittest.TextTestRunner(verbosity=2).run(suite)