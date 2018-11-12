import sys
import pandas as pd

from unittest import TestCase

# Add repository's source files on python inclusion path.
sys.path.insert(0, '../src/')

from predictive_model import DataFrameDataSet


class DataSetTest(TestCase):

    def test_class_distribution_varying_sequence_length(self):
        sequences = [[[{'label': label, 'position': step, 'input': 0, 'sample': str(sequence_length) + '-' + str(label)}
                       for step in range(sequence_length)] for sequence_length in range(1, 10)] for label in [0, 1]]
        classes_distribution_df = DataFrameDataSet(pd.concat([pd.DataFrame(sequences_of_same_length)
                                         for sequences_of_same_length
                                         in (sequences[0] + sequences[1])]),
                              sample_field='sample', step_field='position', label_field='label',
                              test_proportion=0.4, validation_roportion=0.2).classes_distribution(show_chart=False)
        for label_index, label  in enumerate([0, 1]):
            assert classes_distribution_df['num_samples'].iloc[label_index] == 9

    def test_train_test_class_distribution_varying_sequence_length(self):
        sequences = [[[{'label': 0, 'position': step, 'input': 0, 'sample': str(sequence_length) + '-' + str(label)}
                       for step in range(sequence_length)] for sequence_length in range(1, 10)] for label in [0, 1]]
        data = pd.concat([pd.DataFrame(sequences_of_same_length) for sequences_of_same_length in (sequences[0] +
                                                                                                  sequences[1])])
        sequential_data_set = DataFrameDataSet(data, sample_field='sample', step_field='position', label_field='label',
                                               test_proportion=0.4, validation_roportion=0.2)
        data_sets_distribution_df = sequential_data_set.train_test_classes_distribution()
        assert data_sets_distribution_df.values.sum() == len(sequential_data_set)




