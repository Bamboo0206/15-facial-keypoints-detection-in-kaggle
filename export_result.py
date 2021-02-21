"""
use your model to predict the facial
landmark of test.csv, then save the
facial landmark to a file.
"""
import pandas as pd


def export(pred_points, filename):
    """
    :param pred_points: result from your model use test.csv
    :return:
    """
    submission_data = pd.DataFrame(pred_points)
    submission_data.to_csv(filename, index=False)

