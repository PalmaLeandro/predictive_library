import logging
import math
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def init_dir(dir_path, erase_dir=True):
    if tf.gfile.Exists(dir_path) and erase_dir is True:
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)


def plot_dataset(samples, labels=None, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    ax_arg = ax
    ax = ax_arg or plt.subplots(figsize=(10, 10))[1]
    import pandas as pd
    for label_value, label_value_data in pd.DataFrame({'x': samples[0],
                                                       'y': samples[1],
                                                       'label': labels}).groupby('label'):
        label_color = colors.rgb_to_hsv(np.random.rand(3))
        ax.scatter(label_value_data['x'], label_value_data['y'], label=label_value, c=label_color)

    if ax_arg is None:
        plt.show()

def plot_hyperplane(weights, bias, label=None, ax=None, color=None, limits=[-3., 3.]):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    ax_arg = ax
    ax = ax_arg or plt.subplots(figsize=(10, 10))[1]

    color = color or colors.rgb_to_hsv(np.random.rand(3))

    x = np.linspace(limits[0], limits[1])
    y = (weights[:-1] * x + bias) / (- weights[-1])

    # plot normal
    ax.arrow(x[len(x) // 2], y[len(y) // 2], *weights, width=0.05, color=color)

    # plot hyperplane
    ax.plot(x, y, c=color, label=label)
    if label is not None:
        ax.legend()

    if limits is not None:
        ax.set_xlim(limits)
        ax.set_ylim(limits)

    ax.set_aspect('equal', 'box')


def plot_confusion_matrix_heatmap(classifications, labels, classes_labels=None, show_plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    classes_labels = classes_labels \
        if classes_labels is not None \
        else [str(label) for label in np.unique(labels)]
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true=labels,
                                                        y_pred=np.array(classifications)[:, 0].tolist()),
                                       columns=classes_labels, index=classes_labels)
    # Calculate percentages.
    confusion_matrix_df = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)
    # Calculate accuracy.
    confusion_matrix_df['Accuracy'] = [row[row_index] / np.sum([x for x in row])
                                       for row_index, row
                                       in confusion_matrix_df.iterrows()]
    sns.heatmap(confusion_matrix_df, annot=True, ax=plt.subplots(figsize=(18, 10))[1],
                cmap=sns.color_palette("Blues"))
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    if show_plot: plt.show()
    return confusion_matrix_df

sigmoid = lambda x: 1 / (1 + math.exp(-x))

# Vectorized form
v_sigmoid = np.vectorize(sigmoid)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_model_parameters(model_name, session):
    parameters = {}
    trainable_variables = tf.get_variable_scope().trainable_variables()
    for variable in trainable_variables:
        if model_name in variable.name:
            parameters.update({variable.name: session.run(variable)})

    return parameters

def variable_summaries(var, name, add_distribution=True, add_range=True, add_histogram=True):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        if add_distribution:
            real_valued_var = tf.cast(var, tf.float32)
            mean = tf.reduce_mean(real_valued_var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(real_valued_var - mean)))
                tf.summary.scalar('sttdev/' + name, stddev)

        if add_range:
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))

        if add_histogram:
            tf.summary.scalar(name, var)

def linear_neurons_layer(inputs, num_units, scope_name):
    with tf.name_scope(scope_name):
        weights = tf.Variable(tf.truncated_normal(shape=[inputs.get_shape().dims[1].value, num_units],
                                                  stddev=1.0 / math.sqrt(float(num_units))), name='weights')

        biases = tf.Variable(tf.zeros([num_units]), name='biases')

        logits = tf.identity(tf.matmul(inputs, weights) + biases, name='logits')

    return logits


def validate_train_labels(labels):
    if np.max(labels) != len(np.unique(labels)) - 1:
        for label in range(max(np.max(labels), len(np.unique(labels)))):
            if label not in np.unique(labels):
                labels = np.append(labels, np.array([label]))
    return labels


class PredictiveModel(object):

    def __init__(self, num_features=None, batch_size=1,
                 initial_learning_rate=1., learning_rate_decay=0.95, max_epoch_decay=0, session=tf.Session(),
                 log_dir=None, erase_log_dir=True,
                 model_persistence_dir=None, erase_model_persistence_dir=True,
                 is_inner_model=False,
                 **kwargs):
        if not is_inner_model:
            self._session = session
            self._initial_learning_rate = initial_learning_rate
            self._learning_rate_decay = learning_rate_decay
            self._max_epoch_decay = max_epoch_decay
            self._model_persistence_dir = model_persistence_dir
            self._erase_model_persistence_dir = erase_model_persistence_dir

            if model_persistence_dir is not None:
                self.load(model_persistence_dir)
            else:
                inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, *num_features], name='inputs')
                label = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='label')
                self._global_step = tf.Variable(0, name='global_step', trainable=False)
                self._learning_rate = tf.Variable(initial_learning_rate, name='learning_rate', trainable=False)
                self._inference_op = self.build_model(inputs, num_features=num_features,
                                                      batch_size=batch_size, **kwargs)
                self._loss_op = self.build_loss(label, **kwargs)
                self._train_op = self.build_training()
                self._init_summaries(log_dir, erase_log_dir)
                self._session.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

    def _init_summaries(self, log_dir=None, erase_log_dir=True):
        if log_dir is not None:
            self._summaries_op = tf.summary.merge_all()
            init_dir(log_dir, erase_log_dir)
            self._train_summary_writer = tf.summary.FileWriter(log_dir + '/train', self._session.graph)
            self._validation_summary_writer = tf.summary.FileWriter(log_dir + '/validation', self._session.graph)
            self._test_summary_writer = tf.summary.FileWriter(log_dir + '/test', self._session.graph)
        else:
            self._summaries_op = tf.no_op()
            self._train_summary_writer = self._validation_summary_writer = self._test_summary_writer = None

    def build_model(self, inputs, **kwargs):
        '''
        Defines the model that infers an output given the provided input.
        Needs to be implemented by subclasses.
        '''
        return tf.identity(inputs, name=self.name)

    def build_loss(self, label, **kwargs):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._inference_op, labels=label),
                              name='xentropy_mean')


    def build_training(self, max_gradient_norm=5.):
        '''Defines Gradient Descent with clipped norms as default training method.'''
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss_op, tvars), max_gradient_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars))

    # TODO: implement minibatch training with data folding.
    def report_execution(self, samples, labels, operation, summary_writer=None):
        '''Executes an operation while logging the loss and recording the model's summaries'''
        losses = []
        results = []
        for sample_index, (sample, label) in enumerate(zip(samples, labels)):
            feed = {'inputs:0': np.array([sample]), 'label:0': np.array([label])}

            result, loss_value, summaries = self._session.run([operation, self._loss_op, self._summaries_op],
                                                              feed_dict=feed)
            results.append(result)
            losses.append(loss_value)

            if summary_writer is not None:
                summary_writer.add_summary(summaries, global_step=self._session.run(self._global_step))
                summary_writer.flush()

            epoch_completion_perc = float(sample_index / len(samples)) * 100.
            if epoch_completion_perc % 10 == 0:
                logging.info('Epoch completion: {}%'.format(int(epoch_completion_perc)))

        logging.info('Avg. Loss: {}'.format(np.array(losses).mean()))
        return results


    def train(self, samples, labels, num_epochs, validation_size):
        import textwrap
        for epoch in range(num_epochs):
            current_learning_rate_decay = self._learning_rate_decay ** max(epoch - self._max_epoch_decay, 0.0)
            self._session.run(tf.assign(self._learning_rate, self._initial_learning_rate * current_learning_rate_decay))

            # Calculate folds.
            train_samples, validation_samples, \
            train_labels, validation_labels = train_test_split(samples, labels, test_size=validation_size)

            # Update model parameters.
            logging.info(textwrap.dedent('''
            Training.
            Epoch: {} .
            Learning rate: {} .''').format(epoch, self._session.run(self._learning_rate)))
            self.report_execution(samples=train_samples,
                                  labels=validate_train_labels(train_labels),
                                  operation=self._train_op,
                                  summary_writer=self._train_summary_writer)

            # Validate new model's parameters.
            logging.info(textwrap.dedent('''
            Validation.
            Epoch: {} .''').format(epoch))
            self.report_execution(samples=validation_samples,
                                  labels=validation_labels,
                                  operation=tf.no_op(),
                                  summary_writer=self._validation_summary_writer)

            self.save()

    def test(self, samples, labels, classes_labels=None):
        # Test model's performance.
        logging.info('Test.')
        classifications = self.report_execution(samples, labels,
                                                operation=tf.argmax(self._inference_op, axis=1),
                                                summary_writer=self._test_summary_writer)

        confusion_matrix_df = plot_confusion_matrix_heatmap(classifications, labels, classes_labels)
        return confusion_matrix_df['Accuracy'].sum() / len(confusion_matrix_df['Accuracy'])


    def infer(self, inputs):
        return self._session.run(self._inference_op, feed_dict={'inputs:0': inputs})

    def save(self, model_persistence_dir=None, erase_model_persistence_dir=None):
        model_persistence_dir = model_persistence_dir or self._model_persistence_dir
        erase_model_persistence_dir = erase_model_persistence_dir or self._erase_model_persistence_dir
        if model_persistence_dir is not None:
            init_dir(self._model_persistence_dir, erase_model_persistence_dir)
            tf.train.Saver().save(self._session, self._model_persistence_dir + 'model', global_step=self._global_step)

    def load(self, model_persistence_dir=None):
        model_persistence_dir = model_persistence_dir or self._model_persistence_dir
        if model_persistence_dir is not None:
            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(model_persistence_dir) + '.meta')
            saver.restore(self._session, tf.train.latest_checkpoint(model_persistence_dir))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def parameters(self):
        return get_model_parameters(self.name, self._session)

    def plot_parameters(self, samples=None, labels=None, ax_arg=None, limits=[-5.,5.]):
        import matplotlib.pyplot as plt
        ax = ax_arg or plt.subplots(figsize=(10, 10))[1]
        if samples is not None:
            plot_dataset(samples=samples, labels=labels, ax=ax)
        parameters = self.parameters

        hyperplanes_names = np.unique([name.split('/')[0] for name in parameters.keys()]).tolist()
        hyperplanes_weights = [variable_value
                               for variable_name, variable_value
                               in parameters.items()
                               if 'weights' in variable_name]

        hyperplanes_biases = [variable_value
                              for variable_name, variable_value
                              in parameters.items()
                              if 'bias' in variable_name]

        # Assume that the bias and the weights are created at the same time. That's why makes sense the zip.
        for (hyperplanes_names, hyperplanes_weights, hyperplanes_bias) in zip(hyperplanes_names,
                                                                              hyperplanes_weights,
                                                                              hyperplanes_biases):
            for hyperplane_index, \
                (hyperplane_weights, hyperplane_bias) in enumerate(zip(hyperplanes_weights.transpose(),
                                                                       hyperplanes_bias)):
                plot_hyperplane(weights=hyperplane_weights,
                                bias=hyperplane_bias,
                                label=hyperplanes_names + '_hyperplane_' + str(hyperplane_index),
                                ax=ax,
                                limits=limits)

        if ax_arg is None:
            plt.show()


class FastFourierTransform(PredictiveModel):

    def build_model(self, inputs, **kwargs):
        num_steps = inputs.get_shape().dims[1].value
        next_2_base = 2 ** (int(np.log2(num_steps)) + 1)
        return super(FastFourierTransform, self)\
               .build_model(tf.cast(tf.transpose(tf.spectral.rfft(tf.transpose(inputs, [0, 2, 1]),
                                                                  fft_length=[next_2_base]), [0, 2, 1]),
                                    tf.float32))


class BatchReshape(PredictiveModel):

    def build_model(self, inputs, batch_size, new_shape=-1, **kwargs):
        return super(BatchReshape, self).build_model(tf.reshape(inputs, [batch_size, new_shape]))


class SigmoidActivation(PredictiveModel):

    def build_model(self, inputs, **kwargs):
        return super(SigmoidActivation, self).build_model(tf.nn.sigmoid(inputs))


class DeepPredictiveModel(PredictiveModel):
    '''A model that concatenates several PredictiveModels.'''

    def __init__(self, num_features=None, batch_size=None, inner_models_arguments=None, **kwargs):
        # Define if user provided arguments for every model or they would have to share the same.
        inner_models_arguments = inner_models_arguments or np.repeat(kwargs, len(inner_models_arguments))
        for inner_models_argument in inner_models_arguments:
            inner_models_argument['batch_size'] = batch_size
        super(DeepPredictiveModel, self).__init__(num_features, batch_size,
                                                  inner_models_arguments=inner_models_arguments, **kwargs)

    def build_model(self, inputs, inner_models_classes, inner_models_arguments=None, **kwargs):

        current_input = inputs
        current_input_dimension = [int(dimension) for dimension in inputs.get_shape().dims[1:]]
        self.inner_models_names = []
        # Instantiate every model using the provided arguments.
        for inner_model_index, (inner_model_class, inner_model_arguments) in enumerate(zip(inner_models_classes,
                                                                                           inner_models_arguments)):
            inner_model_arguments['num_features'] = current_input_dimension
            inner_model = inner_model_class(is_inner_model=True, **inner_model_arguments)
            inner_model_name = inner_model.name + '_' + str(inner_model_index)
            with tf.variable_scope(inner_model_name):
                current_input = inner_model.build_model(current_input, **inner_model_arguments)
                current_input_dimension = [int(dimension) for dimension in current_input.get_shape().dims[1:]]

            self.inner_models_names.append(inner_model_name)

        return current_input

    @property
    def parameters(self):
        parameters = {}
        for inner_model_name in self.inner_models_names:
            parameters.update(get_model_parameters(inner_model_name, self._session))
        return parameters


class PredictiveSequenceModel(PredictiveModel):
    '''A model that makes inferences over a sequenced input.'''

    def _build_model_evolution(self, input, **kwargs):
        pass

    def reset(self, **kwargs):
        '''
        Reset whatever kind of state the model may have during the prediction along every sequence.
        '''
        pass

    def build_model(self, input_sequence, **kwargs):
        '''
        Executes the model inference over each step of the inputs, evolving the model's internal representation,
        and outputs the final representation as result.
        The input_sequence is assumed to have the following shape:
            [batch_size, num_steps, num_features].
        '''
        result = tf.no_op()
        self.reset(**kwargs)
        for step in range(input_sequence.shape[1]):
            result = self._build_model_evolution(input_sequence[:, step, :], **kwargs)
            # Avoid instantiating all the reusable variables again.
            tf.get_variable_scope().reuse_variables()

        return result


class DeepPredictiveSequenceModel(PredictiveSequenceModel):
    '''A model that concatenates several PredictiveSequenceModels a every timestep of the input sequence.'''

    def reset(self, inner_sequence_models_arguments, **kwargs):
        for inner_sequence_model, inner_sequence_model_arguments in zip(self._inner_sequence_models,
                                                                        inner_sequence_models_arguments):
            inner_sequence_model.reset(**inner_sequence_model_arguments)

    def build_model(self, input_sequence, inner_sequence_models_classes, inner_sequence_models_arguments=None,
                    **kwargs):
        # Define if user provided arguments for every model or they would have to share the same.
        inner_sequence_models_arguments = inner_sequence_models_arguments \
                                          or np.repeat(kwargs, len(inner_sequence_models_classes))

        self.inner_models_names = []
        self._inner_sequence_models = []
        iterator = enumerate(zip(inner_sequence_models_classes, inner_sequence_models_arguments))
        # Instantiate every model in order to use its methods without carrying its class.
        for inner_model_index, (inner_model_class, inner_model_arguments) in iterator:
            inner_model = inner_model_class(is_inner_model=True, **inner_model_arguments)
            self._inner_sequence_models.append(inner_model)
            self.inner_models_names.append(inner_model.name)

        return super(DeepPredictiveSequenceModel,
                     self).build_model(input_sequence,
                                       inner_sequence_models_arguments=inner_sequence_models_arguments,
                                       **kwargs)

    def _build_model_evolution(self, input, inner_sequence_models_arguments, **kwargs):
        current_input = input
        iterator = enumerate(zip(self._inner_sequence_models, inner_sequence_models_arguments))
        for inner_sequence_model_index, (inner_sequence_model, inner_sequence_model_arguments) in iterator:
            with tf.variable_scope(inner_sequence_model.name + '_' + str(inner_sequence_model_index)):
                current_input = inner_sequence_model._build_model_evolution(current_input,
                                                                            **inner_sequence_model_arguments)

        return current_input


class PredictiveRecurrentModel(PredictiveSequenceModel):

    def __init__(self, batch_size, num_units, **kwargs):
        self._num_units = num_units
        self.initial_state = self._build_initial_state(batch_size, self._num_units, **kwargs)
        super(PredictiveSequenceModel, self).__init__(batch_size=batch_size, **kwargs)

    def _build_initial_state(self, batch_size, num_units, **kwargs):
        return tf.zeros([batch_size, num_units])

    def _build_recurrent_model(self, inputs, state, **kwargs):
        '''
        Builds the recurrent model that should update the state and produce an output.
        '''
        pass

    def _build_model_evolution(self, input, **kwargs):
        output, self.state = self._build_recurrent_model(input, self.state, **kwargs)
        return output

    def reset(self, **kwargs):
        self.state = self.initial_state


class LinearNN(PredictiveModel):

    def build_model(self, inputs, num_units, **kwargs):
        return super(LinearNN, self).build_model(linear_neurons_layer(inputs, num_units, self.name))


class Dropout(PredictiveSequenceModel):

    def build_model(self, inputs, keep_prob, **kwargs):
        return tf.nn.dropout(inputs, keep_prob, name='Dropout/output')

    def build_training(self, **kwargs):
        return tf.no_op()

    def build_loss(self, label, **kwargs):
        return tf.no_op()

    def _build_model_evolution(self, inputs, keep_prob, **kwargs):
        return tf.nn.dropout(inputs, keep_prob, name='Dropout/output')


class BasicRNN(PredictiveRecurrentModel):

    def _build_recurrent_model(self, inputs, state, **kwargs):
        current_inputs = inputs
        if inputs.get_shape().dims[1].value != state.get_shape().dims[1].value:
            current_inputs = linear_neurons_layer(inputs, state.get_shape().dims[1].value, 'BasicRNN')

        from rnn_cell import linear
        hidden_state = tf.tanh(linear([current_inputs, state], self._num_units, True, scope='BasicRNN'),
                               'BasicRNN/hidden_state')
        return hidden_state, hidden_state


class IterativeNN(PredictiveModel):

    def build_model(self, input, num_units, max_it, **kwargs):
        weights = tf.get_variable(name='weights', shape=[input.get_shape().dims[1].value, num_units])
        bias = tf.get_variable(name='bias', shape=[num_units])
        weights_norm = tf.norm(weights)

        def iteration_condition(inputs):
            return tf.constant(True)

        def iteration_execution(inputs):
            projection = tf.matmul(inputs, weights) + bias
            activation = tf.nn.sigmoid(projection, name='activation')
            weights_T = tf.transpose(weights)
            return inputs + (1 - activation) * weights_T + activation * (-2) * projection * (weights_T / weights_norm)

        return tf.while_loop(iteration_condition, iteration_execution, [input], maximum_iterations=max_it)