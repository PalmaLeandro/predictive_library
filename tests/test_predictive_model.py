import sys
import pandas as pd

from unittest import TestCase

# Add repository's source files on python inclusion path.
from src.predictive_model import DataFrameDataSet

sys.path.insert(0, '../src/')

from predictive_model import (PredictiveModel,
                              ArrayDataSet,
                              Dropout,
                              LinearNN,
                              SoftmaxActivation,
                              BasicRNN,
                              DeepPredictiveModel,
                              DeepPredictiveSequenceModel)


class PredictiveModelTest(TestCase):

    def test_confusion_matrix(self):
        test_dataset = ArrayDataSet(inputs=[0, 1, 2, 3], labels=[0, 1, 2, 3])
        identity_model = PredictiveModel()
        assert identity_model.test(test_dataset) == 1.

    def test_deep_sequence_predictive_model_within_deep_predictive_model(self):
        sequences = [[[{'label': 0, 'position': step, 'input': 0, 'sample': str(sequence_length) + '-' + str(label)}
                       for step in range(sequence_length)] for sequence_length in range(1, 10)] for label in [0, 1]]
        dataset = DataFrameDataSet(pd.concat([pd.DataFrame(sequences_of_same_length)
                                              for sequences_of_same_length
                                              in (sequences[0] + sequences[1])]),
                                   sample_field='sample', step_field='position', label_field='label',
                                   test_proportion=0.4, validation_roportion=0.2)

        num_layers = 4
        model = DeepPredictiveModel(num_features=[None, dataset.num_features], num_classes=2,
                                    learning_rate_decay=0.6, num_units=20, keep_prob=0.8, batch_size=5,
                                    inner_sequence_models=[Dropout, BasicRNN] * num_layers,
                                    inner_models=[DeepPredictiveSequenceModel,
                                                  (LinearNN, {'num_units': dataset.num_classes}),
                                                  SoftmaxActivation])
        #model.train(dataset, num_epochs=1, batch_size=1)
        model.test(dataset)
