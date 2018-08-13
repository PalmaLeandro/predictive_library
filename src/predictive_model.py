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


def plot_dataset(inputs, labels=None, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    ax_arg = ax
    ax = ax_arg or plt.subplots(figsize=(10, 10))[1]
    import pandas as pd
    for label_value, label_value_data in pd.DataFrame({'x': inputs[0],
                                                       'y': inputs[1],
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


def plot_confusion_matrix_heatmap(predictions, labels, classes_labels=None, show_plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    possible_classes_indices = np.unique(np.array(np.unique(labels).tolist() + np.unique(predictions).tolist()))
    possible_classes_labels = np.array(classes_labels)[possible_classes_indices] if classes_labels is not None \
                              else [str(class_index) for class_index in possible_classes_indices]


    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true=labels.tolist(), y_pred=predictions),
                                       columns=possible_classes_labels, index=possible_classes_labels)

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


def get_current_machine_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    current_machine_ip = s.getsockname()[0]
    s.close()
    return current_machine_ip


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

next_2_base = lambda x: 2 ** (int(np.log2(x)) + 1)

class DistribuibleProgram(object):
    '''An object capable of executing a TensorFlow graph distributed over several machines.'''

    def __init__(self, cluster_machines):
        self._task_index = cluster_machines.index(get_current_machine_ip() + ':0')

        tmp_cluster = tf.train.ClusterSpec({'tmp': cluster_machines})

        self._ps_server = tf.train.Server(tmp_cluster, job_name='tmp', task_index=self._task_index)
        self._worker_server = tf.train.Server(tmp_cluster, job_name='tmp', task_index=self._task_index)

        from tensorflow.core.protobuf import cluster_pb2 as cluster

        cluster_def = cluster.ClusterDef()
        ps_job = cluster_def.job.add()
        ps_job.name = 'ps'
        worker_job = cluster_def.job.add()
        worker_job.name = 'worker'

        for node_index, node in enumerate(cluster_machines):
            ps_job.tasks[node_index] = self._ps_server.target[len('grpc://'):]
            worker_job.tasks[node_index] = self._worker_server.target[len('grpc://'):]

        self._cluster_config = tf.ConfigProto(cluster_def=cluster_def)
        self._cluster = tf.train.ClusterSpec(cluster_def)

        if self._task_index > 0:
            print('Waiting data replicas from ' + cluster_machines[0])

        # # Launch parameter servers.
        # def ps(cluster, task_index):
        #     server = tf.train.Server(cluster, job_name='ps', task_index=task_index)
        #     sess = tf.Session(target=server.target)
        #     sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        #     server.join()
        #
        # ps_process = Process(target=ps, args=(self._cluster, self._task_index))
        # ps_process.start()


class PredictiveModel(DistribuibleProgram):

    def __init__(self, num_features=None,
                 initial_learning_rate=1., learning_rate_decay=0.95, max_epoch_decay=0,
                 cluster_machines=[get_current_machine_ip() + ':0'],
                 log_dir=None, erase_log_dir=True,
                 model_persistence_dir=None, erase_model_persistence_dir=True,
                 is_inner_model=False,
                 **kwargs):
        if not is_inner_model:
            super(PredictiveModel, self).__init__(cluster_machines)
            self._initial_learning_rate = initial_learning_rate
            self._learning_rate_decay = learning_rate_decay
            self._max_epoch_decay = max_epoch_decay
            self._model_persistence_dir = model_persistence_dir
            self._erase_model_persistence_dir = erase_model_persistence_dir

            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(cluster_machines),
                                                                          tf.contrib.training.byte_size_load_fn)
            with tf.device(tf.train.replica_device_setter(self._cluster, ps_strategy=ps_strategy)):
                if model_persistence_dir is not None:
                    self.load(model_persistence_dir)
                else:
                    self._global_step = tf.Variable(0, name='global_step', trainable=False)
                    self._learning_rate = tf.Variable(initial_learning_rate, name='learning_rate', trainable=False)
                    self._update_learning_rate_op = self.build_learning_rate_update(**kwargs)
                    inputs = tf.placeholder(dtype=tf.float32, shape=[None, *num_features], name='inputs')
                    label = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
                    self._do_nothing_op = tf.no_op()
                    self._inference_op = self.build_model(inputs, num_features=num_features, **kwargs)
                    self._infer_class_op = tf.argmax(self._inference_op, axis=1)
                    self._loss_op = self.build_loss(label, **kwargs)
                    self._train_op = self.build_training()
                    self._init_summaries(log_dir, erase_log_dir)
                    # This is required so that tensorflow realizes the true amount of possible classes.
                    self._session.run(self._train_op, {'inputs:0': [np.tile(0, num_features)],
                                                       'label:0': [self._inference_op.shape[-1].value - 1]})

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
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._inference_op,
                                                                             labels=label), name='xentropy_mean')

    def build_training(self, max_gradient_norm=5.):
        '''Defines Gradient Descent with clipped norms as default training method.'''
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss_op, tvars), max_gradient_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars))

    def build_learning_rate_update(self, **kwargs):
        epoch = tf.placeholder(dtype=tf.int32, name='epoch')
        current_learning_rate_decay = self._learning_rate_decay ** \
                                      tf.maximum(tf.cast(epoch, tf.float32) - self._max_epoch_decay, 0.)
        return tf.assign(self._learning_rate, self._initial_learning_rate * current_learning_rate_decay)

    def report_execution(self, inputs, labels, operation, summary_writer=None, batch_size=1, shuffle_samples=False):
        '''Executes an operation while logging the loss and recording the model's summaries'''
        batches_samples_indices = build_samples_indices_batches(num_samples=len(inputs),
                                                                batch_size=batch_size,
                                                                shuffle_samples=shuffle_samples)
        losses = []
        results = []
        for batch_index, batch_samples in enumerate(batches_samples_indices):
            result, loss_value, summaries = self._session.run([operation, self._loss_op, self._summaries_op],
                                                              {'inputs:0': inputs[batch_samples],
                                                               'label:0': labels[batch_samples]})
            results.append(result)
            losses.append(loss_value)

            if summary_writer is not None:
                summary_writer.add_summary(summaries, global_step=self._session.run(self._global_step))
                summary_writer.flush()

            epoch_completion_perc = int(float(batch_index / len(batches_samples_indices)) * 100.)
            if epoch_completion_perc % 10 == 0:
                logging.info('Completion: {}%'.format(epoch_completion_perc))

        logging.info('Avg. Loss: {}'.format(np.array(losses).mean()))
        return results


    def train(self, dataset, num_epochs, validation_size=0.2, batch_size=20):
        import textwrap
        for epoch in range(num_epochs):
            if not self._session.should_stop():
                self._session.run(self._update_learning_rate_op, feed_dict={'epoch:0':epoch})

                # Calculate dataset's folds.
                train_inputs, validation_inputs, \
                train_labels, validation_labels = dataset.provide_train_validation_random_partition(validation_size)

                # Update model's parameters.
                logging.info(textwrap.dedent('''
                Training.
                Epoch: {} .
                Learning rate: {} .''').format(epoch, self._session.run(self._learning_rate)))
                self.report_execution(inputs=train_inputs,
                                      labels=train_labels,
                                      operation=self._train_op,
                                      summary_writer=self._train_summary_writer,
                                      batch_size=batch_size,
                                      shuffle_samples=True)

                # Validate new model's parameters.
                logging.info(textwrap.dedent('''
                Validation.
                Epoch: {} .''').format(epoch))
                self.report_execution(inputs=validation_inputs,
                                      labels=validation_labels,
                                      operation=self._do_nothing_op,
                                      summary_writer=self._validation_summary_writer,
                                      batch_size=max(len(validation_labels) // 10, 10))

                self.save()

    def test(self, dataset):
        # Test model's performance.
        logging.info('Test.')
        classifications = self.report_execution(inputs=dataset.test_inputs,
                                                labels=dataset.test_labels,
                                                operation=self._infer_class_op,
                                                summary_writer=self._test_summary_writer,
                                                batch_size=max(len(dataset.test_labels) // 10, 10))

        confusion_matrix_df = plot_confusion_matrix_heatmap(np.concatenate(classifications),
                                                            dataset.test_labels,
                                                            dataset.label_names)
        return confusion_matrix_df['Accuracy'].sum() / len(confusion_matrix_df['Accuracy'])

    def infer(self, inputs):
        if self._task_index == 0:
            return self._session.run(self._inference_op, feed_dict={'inputs:0': inputs})

    def save(self, model_persistence_dir=None, erase_model_persistence_dir=None):
        if self._task_index == 0:
            model_persistence_dir = model_persistence_dir or self._model_persistence_dir
            erase_model_persistence_dir = erase_model_persistence_dir or self._erase_model_persistence_dir
            if model_persistence_dir is not None:
                init_dir(self._model_persistence_dir, erase_model_persistence_dir)
                tf.train.Saver().save(self._session, self._model_persistence_dir + 'model',
                                      global_step=self._global_step)

    def load(self, model_persistence_dir=None):
        if self._task_index > 0:
            model_persistence_dir = model_persistence_dir or self._model_persistence_dir
            if model_persistence_dir is not None:
                saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(model_persistence_dir) + '.meta')
                saver.restore(self._session, tf.train.latest_checkpoint(model_persistence_dir))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _session(self):
        return tf.train.MonitoredTrainingSession(master=self._worker_server.target,
                                                 is_chief=(self._task_index == 0),
                                                 checkpoint_dir=self._model_persistence_dir,
                                                 config=self._cluster_config)

    @property
    def parameters(self):
        return get_model_parameters(self.name, self._session)

    def plot_parameters(self, inputs=None, labels=None, ax_arg=None, limits=[-5., 5.]):
        import matplotlib.pyplot as plt
        ax = ax_arg or plt.subplots(figsize=(10, 10))[1]
        if inputs is not None:
            plot_dataset(inputs=inputs, labels=labels, ax=ax)
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


class IntermidiateTransformation(PredictiveModel):
    def train(self, samples, labels, num_epochs, validation_size):
        pass

    def test(self, inputs, labels, classes_labels=None):
        pass

    def build_loss(self, label, **kwargs):
        pass

    def build_training(self, max_gradient_norm=5.):
        return self._do_nothing_op


class FastFourierTransform(IntermidiateTransformation):

    def build_model(self, inputs, frame_length, frame_step=None, **kwargs):
        frame_step = frame_step if frame_step is not None else frame_length
        stfts = tf.contrib.signal.stft(inputs, frame_length=frame_length, frame_step=frame_step, pad_end=True)
        return super(FastFourierTransform, self).build_model(tf.squeeze(stfts, axis=2))


class PowerDensityEstimation(FastFourierTransform):

    def build_model(self, inputs, **kwargs):
        stfts = super(PowerDensityEstimation, self).build_model(inputs, **kwargs)
        return super(FastFourierTransform, self).build_model(tf.real(stfts * tf.conj(stfts)), **kwargs)


class LogMagnitudeSpectrogram(FastFourierTransform):

    def build_model(self, inputs, log_offset=1e-6, **kwargs):
        magnitude_spectrograms = tf.abs(super(LogMagnitudeSpectrogram, self).build_model(inputs, **kwargs))
        log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)
        return super(FastFourierTransform, self).build_model(log_magnitude_spectrograms)


class InputTranspose(IntermidiateTransformation):

    def build_model(self, inputs, dimension_placement, **kwargs):
        return super(InputTranspose, self).build_model(tf.transpose(inputs, dimension_placement))


class InputReshape(IntermidiateTransformation):

    def build_model(self, inputs, new_shape, **kwargs):
        return super(InputReshape, self).build_model(tf.reshape(inputs, new_shape))


class BatchReshape(IntermidiateTransformation):

    def build_model(self, inputs, new_shape=[-1], **kwargs):
        return super(BatchReshape, self).build_model(tf.reshape(inputs, [-1, *new_shape]))


class SigmoidActivation(IntermidiateTransformation):

    def build_model(self, inputs, **kwargs):
        return super(SigmoidActivation, self).build_model(tf.nn.sigmoid(inputs))


class DeepPredictiveModel(PredictiveModel):
    '''A model that concatenates several PredictiveModels.'''

    def __init__(self, num_features=None, inner_models_arguments=None, **kwargs):
        # Define if user provided arguments for every model or they would have to share the same.
        inner_models_arguments = inner_models_arguments or np.repeat(kwargs, len(inner_models_arguments))
        super(DeepPredictiveModel, self).__init__(num_features, inner_models_arguments=inner_models_arguments, **kwargs)

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

class TransformationPipeline(DeepPredictiveModel):

    def train(self, samples, labels, num_epochs, validation_size):
        pass

    def test(self, inputs, labels, classes_labels=None):
        pass

    def build_training(self, max_gradient_norm=5.):
        pass

    def build_loss(self, label, **kwargs):
        pass


class PredictiveSequenceModel(PredictiveModel):
    '''A model that makes inferences over a sequenced input.'''

    def build_model_evolution(self, input, **kwargs):
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
            result = self.build_model_evolution(input_sequence[:, step, :], **kwargs)
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


    def build_model_evolution(self, input, inner_sequence_models_arguments, **kwargs):
        current_input = input
        iterator = enumerate(zip(self._inner_sequence_models, inner_sequence_models_arguments))
        for inner_sequence_model_index, (inner_sequence_model, inner_sequence_model_arguments) in iterator:
            with tf.variable_scope(inner_sequence_model.name + '_' + str(inner_sequence_model_index)):
                current_input = inner_sequence_model.build_model_evolution(current_input,
                                                                            **inner_sequence_model_arguments)

        return current_input


class PredictiveRecurrentModel(PredictiveSequenceModel):

    def __init__(self, num_units, **kwargs):
        self._num_units = num_units
        self.initial_state = self._build_initial_state(self._num_units, **kwargs)
        super(PredictiveSequenceModel, self).__init__(**kwargs)

    def _build_initial_state(self, batch_size, num_units, **kwargs):
        return tf.zeros([batch_size, num_units])

    def _build_recurrent_model(self, inputs, state, **kwargs):
        '''
        Builds the recurrent model that should update the state and produce an output.
        '''
        pass

    def build_model_evolution(self, input, **kwargs):
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
        return self._do_nothing_op

    def build_loss(self, label, **kwargs):
        return self._do_nothing_op

    def build_model_evolution(self, inputs, keep_prob, **kwargs):
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


class DataSet(object):

    @property
    def label_names(self):
        pass

    def get_all_samples(self):
        pass

    def get_sample(self):
        pass

    def get_sample_batch(self, batch_size):
        pass

    def provide_train_validation_random_partition(self):
        pass


def dataset_class_distribution(labels, labels_names=None, show_table=True, plot_chart=True, ax=None, **kwargs):
    classes_histogram = np.unique(labels.tolist(), return_counts=True)
    if not show_table and not plot_chart:
        return classes_histogram
    import pandas as pd
    labels_names = labels_names or [str(class_index) for class_index in classes_histogram[0]]
    classes_histogram_df = pd.DataFrame({'class': [labels_names[class_index]
                                                   for class_index, class_num_samples
                                                   in zip(*classes_histogram)],
                                         'num_samples': [class_num_samples
                                                         for class_index, class_num_samples
                                                         in zip(*classes_histogram)]})
    if plot_chart:
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.subplot(**kwargs)

        ax.bar(classes_histogram_df['class'], classes_histogram_df['num_samples'])

    if show_table:
        classes_histogram_df.set_index('class', inplace=True)

    return classes_histogram_df, ax


def sample_inputs_labels(inputs, labels=None):
    import random
    sample_index = random.randrange(len(inputs))
    if labels is not None:
        assert len(labels) == len(inputs), 'Inputs an labels lengths don\'t match.'
        return inputs[sample_index], labels[sample_index]
    else:
        return inputs[sample_index]


def build_samples_indices_batches(num_samples, batch_size=1, shuffle_samples=False):
    assert batch_size <= num_samples, 'The batch requested is too large.'
    num_batches = num_samples // batch_size
    samples_indices = list(range(num_samples))
    if shuffle_samples:
        import random
        samples_indices = random.sample(samples_indices, k=num_samples)
    samples_indices_batches = [samples_indices[batch_index * batch_size: batch_index * batch_size + batch_size]
                               for batch_index in range(num_batches)]
    if num_samples % batch_size == 0:
        return samples_indices_batches
    else:
        return samples_indices_batches + [samples_indices[num_batches * batch_size:]] # Batches' leftover.


class ArrayDataSet(DataSet):

    def __init__(self, inputs, labels=np.array([]),
                 test_proportion=None, validation_proportion=None, labels_names=None, **kwargs):
        self.labels_names = labels_names
        if test_proportion is None:
            self.train_inputs, self.train_labels = inputs, labels
            self.test_inputs, self.test_labels = inputs, labels
        else:
            self.train_inputs, self.test_inputs, self.train_labels, self.test_labels = train_test_split(inputs,
                                                                                                        labels,
                                                                                        test_size=test_proportion)
            self.validation_proportion = validation_proportion

    @property
    def label_names(self):
        return self.labels_names if self.labels_names is not None else [str(label) for label in range(self.num_classes)]

    @property
    def num_samples(self):
        return len(self.get_all_samples()[0])

    def __len__(self):
        return self.num_samples

    @property
    def num_train_samples(self):
        return len(self.train_inputs)

    @property
    def num_test_samples(self):
        return len(self.train_inputs)

    @property
    def num_classes(self):
        # Consider labels as indices being counted from zero.
        return self.train_labels.max() + 1

    @property
    def num_features(self):
        sample_inputs = self.get_sample()[0]
        # If the sample is a sequence then return the length of the first step.
        return len(sample_inputs if len(sample_inputs.shape) == 1 else sample_inputs[0])

    def provide_train_validation_random_partition(self, validation_proportion=None):
        return train_test_split(self.train_inputs, self.train_labels,
                                test_size=validation_proportion or self.validation_proportion)

    def get_all_samples(self):
        return np.concatenate([self.train_inputs, self.test_inputs]), \
               np.concatenate([self.train_labels, self.test_labels])

    def get_sample(self):
        return sample_inputs_labels(*self.get_all_samples())

    def get_train_sample(self):
        return sample_inputs_labels(self.train_inputs, self.train_labels)

    def get_test_sample(self):
        return sample_inputs_labels(self.test_inputs, self.test_labels)

    def num_batches(self, batch_size):
        return self.num_samples // batch_size

    def num_train_batches(self, batch_size):
        return self.num_train_samples // batch_size

    def get_train_sample_batch(self, batch_size):
        return build_samples_indices_batches(self.train_inputs, self.train_labels, batch_size)

    def classes_distribution(self, show_chart=True):
        classes_histogram_df, ax = dataset_class_distribution(self.get_all_samples()[1], self.labels_names,
                                                              title='Dataset classes distribution')
        if show_chart:
            import matplotlib.pyplot as plt
            plt.show()
            return classes_histogram_df, ax
        else:
            return classes_histogram_df

    def train_test_classes_distribution(self):
        import pandas as pd
        import matplotlib.pyplot as plt

        _, (train_dataset_classes_ax, test_dataset_classes_ax) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

        train_samples_histogram_df, _ = dataset_class_distribution(self.train_labels, self.labels_names,
                                                                   show_table=False,
                                                                   ax=train_dataset_classes_ax)
        plt.sca(train_dataset_classes_ax)
        plt.xticks(rotation=-90)
        train_dataset_classes_ax.set_title('Train dataset classes distribution')

        test_samples_histogram_df, _ = dataset_class_distribution(self.test_labels, self.labels_names,
                                                                  show_table=False,
                                                                  ax=test_dataset_classes_ax)

        plt.sca(test_dataset_classes_ax)
        plt.xticks(rotation=-90)
        test_dataset_classes_ax.set_title('Test dataset classes distribution')

        train_samples_histogram_df['dataset'] = 'Train'
        test_samples_histogram_df['dataset'] = 'Test'
        return pd.concat([train_samples_histogram_df, test_samples_histogram_df]).pivot(index='class',
                                                                                        columns='dataset')


class CSVDataSet(ArrayDataSet):

    def __init__(self, path, sample_field=None, step_field=None, label_field=None,
                 test_proportion=None, validation_proportion=None, labels_names=None, **kwargs):
        import pandas as pd
        dataset_df = pd.read_csv(path, **kwargs)

        if sample_field is not None and sample_field in dataset_df.columns and \
           label_field is not None and label_field in dataset_df.columns:
            num_samples = dataset_df[sample_field].nunique()

            labels = dataset_df[[sample_field,
                                 label_field]].drop_duplicates().set_index(sample_field)[label_field].as_matrix()

            if step_field is not None and step_field in dataset_df.columns:
                # Treatment for sequential datasets.
                self.steps_per_sample = dataset_df[step_field].nunique()
                num_features = len([column for column in dataset_df.columns if column not in [sample_field,
                                                                                              step_field,
                                                                                              label_field]])

                inputs = dataset_df.drop(label_field, axis=1).set_index([sample_field, step_field]).as_matrix()\
                            .reshape([num_samples, self.steps_per_sample, num_features])
            else:
                self.steps_per_sample = None
                inputs = dataset_df.drop(label_field, axis=1).set_index(sample_field).as_matrix()

        else:
            self.steps_per_sample = None
            if label_field is not None and label_field in dataset_df.columns:
                inputs = dataset_df.drop(label_field, axis=1).as_matrix()
                labels = dataset_df[label_field].as_matrix()
            else:
                inputs = dataset_df.as_matrix()
                labels = np.array([])

        super(CSVDataSet, self).__init__(inputs, labels,
                                         test_proportion=test_proportion, validation_proportion=validation_proportion,
                                         labels_names=labels_names, **kwargs)

def list_folder_files_with_extension(folder_path, extension):
    import glob
    folder_path = folder_path + '*' if folder_path[-1] == '/' else '/*'
    return [filename for filename in glob.glob(folder_path) if filename.split('.')[-1] == extension]


class SequentialData(ArrayDataSet):

    @property
    def num_samples(self):
        pass

    @property
    def num_features(self):
        pass

    @property
    def steps_per_sample(self):
        pass

    @property
    def shape(self):
        return [self.num_samples, self.steps_per_sample, self.num_features]

    def __getitem__(self, key):
        pass


class PartialSequentialData(SequentialData):
    """Class that wraps a SequentialData in order to retrieve only part of it, reusing its main functionality.
       This is particularly useful to retrieve inputs or labels separately to build an ArrayDataSet."""

    def __init__(self, complete_sequential_data):
        self.complete_sequential_data = complete_sequential_data

    @property
    def num_samples(self):
        return self.complete_sequential_data.num_samples

    @property
    def num_features(self):
        return self.complete_sequential_data.num_features

    @property
    def steps_per_sample(self):
        return self.complete_sequential_data.steps_per_sample

    @property
    def num_classes(self):
        return self.complete_sequential_data.num_classes

    def __getitem__(self, key):
        """Implement acquisition of the part of the complete data that will be retrieved."""
        pass


class KeyPartialSequentialData(PartialSequentialData):

    def __init__(self, data_key, **kwargs):
        self.data_key = data_key
        super(KeyPartialSequentialData, self).__init__(**kwargs)

    def __getitem__(self, key):
        return self.complete_sequential_data[key][self.data_key]

    def get_all_samples(self):
        return self.complete_sequential_data[:, self.data_key]

    def tolist(self):
        return self.complete_sequential_data[:, self.data_key]


def data_partition(data_sequence_sets, key):
    return SequentialDataMerge([data_sequence_set[:, key] for data_sequence_set in data_sequence_sets])


def validate_train_labels(labels):
    labels = labels.tolist()
    if np.max(labels) != len(np.unique(labels)) - 1:
        for label in range(max(np.max(labels), len(np.unique(labels)))):
            if label not in np.unique(labels):
                labels = np.append(labels, np.array([label]))
    return labels


class SequentialDataMerge(SequentialData):

    def __init__(self, data_sequences_sets, test_proportion=None, validation_proportion=None, labels_names=None,
                 exact_merge=False, inputs_key=0, labels_key=1, **kwargs):
        self.data_sequences_sets = data_sequences_sets
        self.exact_merge = exact_merge

        if exact_merge:
            super(SequentialData, self).__init__(inputs=KeyPartialSequentialData(inputs_key,
                                                                                 complete_sequential_data=self),
                                                 labels=KeyPartialSequentialData(labels_key,
                                                                                 complete_sequential_data=self),
                                                 test_proportion=test_proportion,
                                                 validation_proportion=validation_proportion,
                                                 labels_names=labels_names, **kwargs)
        else:
            if test_proportion not in (0, 1, None):
                self.num_train_sequence_sets = int(round(len(data_sequences_sets) * (1 - test_proportion)))
                self.num_test_sequence_sets = int(round(len(data_sequences_sets) * (test_proportion)))
                assert self.num_train_sequence_sets + self.num_test_sequence_sets == len(self.data_sequences_sets)
                self.test_sequence_sets = self.data_sequences_sets[self.num_train_sequence_sets:]
                self.inputs_key = inputs_key
                self.labels_key = labels_key
                self.labels_names = labels_names

    @property
    def train_inputs(self):
        return data_partition(self.data_sequences_sets[:self.num_train_sequence_sets], self.inputs_key)

    @property
    def train_labels(self):
        return data_partition(self.data_sequences_sets[:self.num_train_sequence_sets], self.labels_key)
    @property
    def test_inputs(self):
        return data_partition(self.data_sequences_sets[self.num_train_sequence_sets:], self.inputs_key)

    @property
    def test_labels(self):
        return data_partition(self.data_sequences_sets[self.num_train_sequence_sets:], self.labels_key)

    def provide_train_validation_random_partition(self, validation_proportion=None):
        validation_proportion = validation_proportion or self.validation_proportion
        if self.exact_merge:
            return super(SequentialDataMerge, self).provide_train_validation_random_partition(validation_proportion)
        else:
            # Here we go.
            import random
            train_sequence_sets = self.data_sequences_sets[:self.num_train_sequence_sets]
            number_of_sequence_sets_on_train_fold = int(round(len(train_sequence_sets) * (1 - validation_proportion)))

            train_sequence_sets_indices_fold = random.sample(range(len(train_sequence_sets)),
                                                             k=number_of_sequence_sets_on_train_fold)
            train_sequence_sets_fold = [train_sequence_sets[data_sequence_set_index]
                                        for data_sequence_set_index in train_sequence_sets_indices_fold]

            validation_sequence_sets_indices_fold = list(set(range(number_of_sequence_sets_on_train_fold)) - \
                                                         set(train_sequence_sets_indices_fold))
            validation_sequence_sets_fold = [train_sequence_sets[data_sequence_set_index]
                                             for data_sequence_set_index in validation_sequence_sets_indices_fold]

            fold_train_inputs = SequentialDataMerge([data_sequence_set[:, self.inputs_key]
                                                    for data_sequence_set in train_sequence_sets_fold],
                                                    test_proportion=0, validation_proportion=0,
                                                    labels_names=self.labels_names, exact_merge=False,
                                                    inputs_key=self.inputs_key, labels_key=self.labels_key)
            fold_train_labels = SequentialDataMerge([data_sequence_set[:, self.labels_key]
                                                     for data_sequence_set in train_sequence_sets_fold],
                                                    test_proportion=1, validation_proportion=0,
                                                    labels_names=self.labels_names, exact_merge=False,
                                                    inputs_key=self.inputs_key, labels_key=self.labels_key)

            fold_validation_inputs = SequentialDataMerge([data_sequence_set[:, self.inputs_key]
                                                          for data_sequence_set in validation_sequence_sets_fold],
                                                         test_proportion=0, validation_proportion=1,
                                                         labels_names=self.labels_names, exact_merge=False,
                                                         inputs_key=self.inputs_key, labels_key=self.labels_key)
            fold_validation_labels = SequentialDataMerge([data_sequence_set[:, self.labels_key]
                                                          for data_sequence_set in validation_sequence_sets_fold],
                                                         test_proportion=1, validation_proportion=1,
                                                         labels_names=self.labels_names, exact_merge=False,
                                                         inputs_key=self.inputs_key, labels_key=self.labels_key)

            return fold_train_inputs, fold_validation_inputs, fold_train_labels, fold_validation_labels


    @property
    def num_samples(self):
        return sum([len(data_sequences_set) for data_sequences_set in self.data_sequences_sets])

    @property
    def num_train_samples(self):
        return sum([data_sequences_set.num_samples
                    for data_sequences_set
                    in self.data_sequences_sets[:self.num_train_sequence_sets]])

    @property
    def num_test_samples(self):
        return sum([data_sequences_set.num_samples
                    for data_sequences_set
                    in self.data_sequences_sets[self.num_train_sequence_sets:]])

    @property
    def num_features(self):
        return self.data_sequences_sets[0].num_features

    @property
    def steps_per_sample(self):
        return self.data_sequences_sets[0].steps_per_sample

    @property
    def num_classes(self):
        return max(*[data_sequences_set.num_classes for data_sequences_set in self.data_sequences_sets])

    def sample_at(self, index):
        if index < self.num_samples:
            datasets_limits_cumsum = np.cumsum([len(data_sequences_set)
                                                for data_sequences_set
                                                in self.data_sequences_sets]).tolist()
            data_sequences_set_index, sample_index = [(data_sequences_set_index, index - dataset_start_index)
                                                      for data_sequences_set_index, (dataset_start_index,
                                                                                     dataset_end_index)
                                                      in enumerate(zip([0] + datasets_limits_cumsum,
                                                                       datasets_limits_cumsum + [0]))
                                                      if index < dataset_end_index][0]
            return self.data_sequences_sets[data_sequences_set_index][sample_index]
        else:
            raise IndexError('Requested sample at {} is out of bounds.'.format(index))

    def __getitem__(self, key):
        if np.isscalar(key):
            return self.sample_at(key)
        else:
            return [self.sample_at(index) for index in np.array(range(self.num_samples))[key]]

    def tolist(self):
        return [self.sample_at(sample_index) for sample_index in range(self.num_samples)]


class EpochEegExperimentData(SequentialData):

    eeg_signal_sample_position = 0
    label_sample_position = 1

    def __init__(self, files_folder_path, epoch_duration):
        self.files_folder_path = files_folder_path
        self.epoch_duration = epoch_duration

        # If there is a next onset(to compare with) and it's steps_per_sample ahead, then is a valid epoch.
        signal_classification = self.signal_classification
        steps_per_sample = self.steps_per_sample
        self.valid_samples = [(onset, label) for onset_index, (onset, label) in enumerate(signal_classification)
                              if onset_index + 1 < len(signal_classification)
                              and (signal_classification[onset_index + 1][0] - onset) == steps_per_sample]

    @property
    def eeg_signals(self):
        import mne
        set_filename = list_folder_files_with_extension(self.files_folder_path, 'set')[0]
        return mne.io.read_raw_eeglab(input_fname=set_filename, preload=False)

    @property
    def signal_classification(self):
        pass

    @property
    def num_samples(self):
        return len(self.valid_samples)

    @property
    def steps_per_sample(self):
        return int(self.eeg_signals.info.get('sfreq')) * self.epoch_duration

    @property
    def num_features(self):
        import mne
        return len(mne.pick_types(self.eeg_signals.info, eeg=True, stim=False))

    @property
    def num_classes(self):
        # Plus one since classes are considerated to be enumerated from 0.
        return np.array(self.valid_samples)[:, self.label_sample_position].max() + 1

    def __getitem__(self, key):
        if np.isscalar(key):
            onset, label = self.valid_samples[key]
            # Omit empty stim channel and time dimmension.
            return np.transpose(self.eeg_signals[:-1, onset:onset + self.steps_per_sample][0]), label
        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                if key[1] == self.label_sample_position:
                    return np.array(self.valid_samples)[key]
                if key[1] == self.eeg_signal_sample_position:
                    return KeyPartialSequentialData(data_key=self.eeg_signal_sample_position,
                                                    complete_sequential_data=self)


class FmedLfaEegExperimentData(EpochEegExperimentData):

    @property
    def signal_classification(self):
        import scipy.io
        mat_filename = list_folder_files_with_extension(self.files_folder_path, 'mat')[0]
        labels_data = scipy.io.loadmat(mat_filename)['stageData'][0][0]

        # Select the index of the stages in de matrix, most of the time is 5 but sometimes it's 6.
        stage_info_index = 5 if labels_data[5].dtype == np.dtype('uint8') else 6
        onset_index = stage_info_index + 1

        return list(zip(labels_data[onset_index][:,0], labels_data[stage_info_index][:,0]))


class FmedLfaExperimentDataSet(SequentialDataMerge):

    def __init__(self, experiments_data_folders, epoch_duration, test_proportion, validation_proportion, labels_names):
        super(FmedLfaExperimentDataSet, self).__init__([FmedLfaEegExperimentData(experiments_data_folder,
                                                                                 epoch_duration)
                                                        for experiments_data_folder in experiments_data_folders],
                                                        test_proportion, validation_proportion, labels_names)