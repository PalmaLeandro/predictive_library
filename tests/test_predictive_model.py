import sys

from unittest import TestCase

# Add repository's source files on python inclusion path.
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
        num_layers = 4
        model = DeepPredictiveModel(num_features=[None, 150], learning_rate_decay=0.6, num_units=20, keep_prob=0.8,
                                    batch_size=1,
                                    inner_sequence_models=[Dropout, BasicRNN] * num_layers,
                                    inner_models=[DeepPredictiveSequenceModel,
                                                  (LinearNN, {'num_units': 2}),
                                                  SoftmaxActivation])
