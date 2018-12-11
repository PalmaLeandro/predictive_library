import logging
import math
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.ops import array_ops

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


def plot_hyperplane(weights, bias, label=None, ax=None, color=None, limits=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    limits = limits if limits is not None else [-3., 3.]

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

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true=np.array(labels).tolist(), y_pred=predictions),
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


def plot_hyperplanes(parameters, limits=None, ax=None):
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


def plot_learning(parameters, inputs=None, labels=None, ax_arg=None, limits=[-5., 5.]):
    import matplotlib.pyplot as plt
    ax = ax_arg or plt.subplots(figsize=(10, 10))[1]
    if inputs is not None:
        plot_dataset(inputs=inputs, labels=labels, ax=ax)

    plot_hyperplanes(parameters, limits, ax_arg, ax)


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    from IPython.display import display, HTML
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


class DistribuibleProgram(object):
    '''An object capable of executing a TensorFlow graph distributed over several machines.'''

    def __init__(self, cluster_machines):
        self._task_index = cluster_machines.index(get_current_machine_ip() + ':0')

        tmp_cluster = tf.train.ClusterSpec({'tmp': cluster_machines}) #

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

    def __init__(self, num_features=[None], batch_size=None, num_units=None,
                 cluster_machines=[get_current_machine_ip() + ':0'],
                 model_persistence_dir=None, erase_model_persistence_dir=True,
                 is_inner_model=False, **kwargs):
        if not is_inner_model:
            super().__init__(cluster_machines)
            self._model_persistence_dir = model_persistence_dir
            self._erase_model_persistence_dir = erase_model_persistence_dir

            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(cluster_machines),
                                                                          tf.contrib.training.byte_size_load_fn)
            with tf.device(tf.train.replica_device_setter(self._cluster, ps_strategy=ps_strategy)):
                if model_persistence_dir is not None:
                    self.load(model_persistence_dir)
                else:
                    self._batch_size = batch_size
                    self._global_step = tf.train.get_or_create_global_step()

                    inputs = tf.placeholder(dtype=tf.float32, shape=[None, *num_features], name='inputs')
                    label = tf.placeholder(dtype=tf.float32, shape=[None], name='label')

                    self._do_nothing_op = tf.no_op()
                    self._model_output_op = self.build_model(inputs, num_features=num_features, num_units=num_units,
                                                             **kwargs)
                    self._inference_op = self.build_inference(self._model_output_op, **kwargs)
                    self._label_op = self.build_label(label, **kwargs)
                    self._loss_op = self.build_loss(self._label_op, self._model_output_op, **kwargs)
                    self._summaries_op = self.build_summaries()
                    self._global_initializer = tf.global_variables_initializer()

            self._session = tf.train.MonitoredTrainingSession(master=self._worker_server.target,
                                                              is_chief=(self._task_index == 0),
                                                              checkpoint_dir=self._model_persistence_dir,
                                                              config=self._cluster_config)
            self._session.run(self._global_initializer)

        self.output_size = num_units

    def build_model(self, inputs, name=None, **kwargs):
        '''
        Defines the model output to be used to infer a prediction, given the provided input.
        Needs to be implemented by subclasses.
        '''
        output = tf.identity(inputs, name=name or (self.name + '_inference'))
        self.output_size = tf.shape(inputs)[1:]
        return output

    def build_inference(self, model_output, name=None, **kwargs):
        '''
        Defines the usage of the output of the model to produce a prediction, given the provided input.
        '''
        return tf.identity(model_output, name=name or self.name)

    def build_label(self, label, **kwargs):
        return tf.identity(label, name='label')

    def build_loss(self, label, prediction, **kwargs):
        return tf.reduce_mean(tf.squared_difference(label, prediction, name='squared_error'), name='mean_squared_error')

    def report_execution(self, inputs, labels, operation, summary_writer=None, batch_size=1, shuffle_samples=False,
                         return_batches_samples_indices=False, persist_to_path=None, **op_args):
        '''Executes an operation while logging the loss and recording the model's summaries.'''
        batches_samples_indices = build_samples_indices_batches(len(inputs), batch_size, shuffle_samples)
        num_batches = len(batches_samples_indices)

        if persist_to_path is None:
            results = []
        else:
            expected_results_shape = (num_batches, batch_size) + \
                                     tuple([dim.value for dim in operation.get_shape().dims if dim.value is not None])
            results = np.memmap(filename=persist_to_path + self.name, dtype=np.float32, mode='w+',
                                shape=expected_results_shape)
        losses = []
        for batch_index, batch_samples in enumerate(batches_samples_indices):
            graph_args = {**{key + ':0': value for key, value in op_args.items()},
                          **{'inputs:0': [inputs[sample] for sample in batch_samples],
                             'label:0': [labels[sample] for sample in batch_samples]}}
            result, loss_value, summaries = self._session.run([operation, self._loss_op, self._summaries_op],
                                                              feed_dict=graph_args)

            if persist_to_path is None:
                results.append(result)
            else:
                if result.shape[0] < batch_size:
                    # This is a batch leftover, we need to add samples to fill the result.
                    results[batch_index] = np.concatenate([result, np.array([np.zeros(result.shape[1:])
                                                                             for artificial_sample
                                                                             in range(batch_size - result.shape[0])])])
                else:
                    # All good.
                    results[batch_index] = result

            losses.append(loss_value)

            if batch_index in range(0, num_batches, math.ceil(num_batches * .1)):
                logging.info('Completion: {}%'.format((batch_index * 100) // num_batches))

        logging.info('Avg. Loss: {}'.format(np.array(losses).mean()))
        return results if return_batches_samples_indices is False else (results, batches_samples_indices)

    def test(self, data_set):
        # Test model's performance.
        logging.info('Test.')
        return self.report_execution(inputs=data_set.test_inputs, labels=data_set.test_labels,
                                     operation=self._inference_op, summary_writer=self._test_summary_writer,
                                     batch_size=max(len(data_set.test_labels) // 10, 1))

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

    def build_summaries(self, **kwargs):
        return self._do_nothing_op

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self, **kwargs):
        '''Reset whatever kind of state the model may have during the prediction if it is done over a sequence.'''
        pass

    def build_model_evolution(self, inputs, **kwargs):
        return self.build_model(inputs, **kwargs)


class ClassificationModel(PredictiveModel):

    def build_inference(self, model_output, name=None, **kwargs):
        return super().build_inference(tf.argmax(tf.nn.softmax(model_output), axis=1), name='prediction', **kwargs)

    def build_label(self, label, num_classes=None, risk_function='mean_squared_error', **kwargs):
        if risk_function == 'cross_entropy':
            return super(ClassificationModel, self).build_label(tf.cast(label, tf.int64))
        else:
            return super(ClassificationModel, self).build_label(tf.one_hot(tf.cast(label, tf.int64), num_classes))

    def build_loss(self, label, prediction, risk_function='mean_squared_error', **kwargs):
        if risk_function == 'cross_entropy':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                                                 labels=label,
                                                                                 name='cross_entropy'),
                                  name='cross_entropy_mean')
        else:
            return super().build_loss(label, prediction)

    def test(self, data_set):
        classifications = super().test(data_set)
        confusion_matrix_df = plot_confusion_matrix_heatmap(np.concatenate(classifications),
                                                            data_set.test_labels,
                                                            data_set.label_names)
        return confusion_matrix_df['Accuracy'].sum() / len(confusion_matrix_df['Accuracy'])


class TrainableModel(ClassificationModel):

    def build_loss(self, label, prediction, initial_learning_rate=1., **kwargs):
        loss_op = super().build_loss(label, prediction, **kwargs)
        self._learning_rate = tf.Variable(initial_learning_rate, name='learning_rate', trainable=False)
        self._update_learning_rate_op = self.build_learning_rate_update(initial_learning_rate=initial_learning_rate,
                                                                        **kwargs)
        self._train_op = self.build_training(loss_op, **kwargs)
        self.build_summaries(**kwargs)
        return loss_op

    def build_training(self, loss, max_gradient_norm=5., **kwargs):
        '''Defines Gradient Descent with clipped norms as default training method.'''
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_gradient_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars))

    def build_learning_rate_update(self, initial_learning_rate=1., learning_rate_decay=.9, max_epoch_decay=1, **kwargs):
        epoch = tf.placeholder(shape=(), dtype=tf.float32, name='epoch')
        current_learning_rate_decay = learning_rate_decay ** tf.maximum(epoch - max_epoch_decay, 0.)
        self._learning_rate = tf.assign(self._learning_rate, initial_learning_rate * current_learning_rate_decay)
        return self._learning_rate

    def build_summaries(self, log_dir=None, erase_log_dir=False, **kwargs):
        if log_dir is not None:
            init_dir(log_dir, erase_log_dir)
            self._train_summary_writer = tf.summary.FileWriter(log_dir + '/train', self._session.graph)
            self._validation_summary_writer = tf.summary.FileWriter(log_dir + '/validation', self._session.graph)
            self._test_summary_writer = tf.summary.FileWriter(log_dir + '/test', self._session.graph)
            return tf.summary.merge_all()
        else:
            self._train_summary_writer = self._validation_summary_writer = self._test_summary_writer = None
            return super().build_summaries(**kwargs)

    def train(self, dataset, num_epochs=1, validation_size=0., batch_size=1):
        import textwrap
        for epoch in range(num_epochs):
            #if not self._session.should_stop():

                # Calculate dataset's folds.
            train_inputs, validation_inputs, \
            train_labels, validation_labels = dataset.provide_train_validation_random_partition(validation_size)

            # Update model's parameters.
            logging.info(textwrap.dedent('''
            Training.
            Epoch: {} .
            Learning rate: {} .''').format(epoch, self._session.run(self._learning_rate, {'epoch:0':epoch})))
            self.report_execution(inputs=train_inputs,
                                  labels=train_labels,
                                  operation=self._train_op,
                                  summary_writer=self._train_summary_writer,
                                  batch_size=min(batch_size, len(train_inputs)),
                                  shuffle_samples=True,
                                  epoch=epoch)

            # Validate new model's parameters.
            logging.info(textwrap.dedent('''
            Validation.
            Epoch: {} .''').format(epoch))
            self.report_execution(inputs=validation_inputs,
                                  labels=validation_labels,
                                  operation=self._do_nothing_op,
                                  summary_writer=self._validation_summary_writer,
                                  batch_size=min(batch_size, len(validation_labels)))

            self.save()

    @property
    def parameters(self):
        return get_model_parameters(self.name, self._session)


class FastFourierTransform(PredictiveModel):

    def build_model(self, inputs, frame_length, frame_step=None, **kwargs):
        frame_step = frame_step if frame_step is not None else frame_length
        stfts = tf.contrib.signal.stft(inputs, frame_length=frame_length, frame_step=frame_step, pad_end=True)
        return super().build_model(stfts)


class PowerDensityEstimation(FastFourierTransform):

    def build_model(self, inputs, **kwargs):
        stfts = super(PowerDensityEstimation, self).build_model(inputs, **kwargs)
        return super(FastFourierTransform, self).build_model(tf.real(stfts * tf.conj(stfts)), **kwargs)


class LogMagnitudeSpectrogram(FastFourierTransform):

    def build_model(self, inputs, log_offset=1e-6, **kwargs):
        magnitude_spectrograms = tf.abs(super(LogMagnitudeSpectrogram, self).build_model(inputs, **kwargs))
        log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)
        return super(FastFourierTransform, self).build_model(log_magnitude_spectrograms)


class InputTranspose(PredictiveModel):

    def build_model(self, inputs, dimension_placement, **kwargs):
        return super().build_model(tf.transpose(inputs, dimension_placement))


class InputReshape(PredictiveModel):

    def build_model(self, inputs, new_shape, **kwargs):
        return super().build_model(tf.reshape(inputs, new_shape))


class BatchReshape(PredictiveModel):

    def build_model(self, inputs, new_shape=[-1], **kwargs):
        return super().build_model(tf.reshape(inputs, [-1, *new_shape]))


class BatchSlice(PredictiveModel):

    def build_model(self, inputs, start=None, stop=None, **kwargs):
        input_slice = tuple(np.repeat(slice(None, None, None), len(inputs.get_shape()) - 1).tolist() +
                              [slice(start, stop, None)])
        return super().build_model(inputs[input_slice])


class BatchAggregation(PredictiveModel):

    def build_model(self, inputs, aggregation, dimensions=None, **kwargs):
        if dimensions is None:
            # not_batch_dimensions_indexes:
            dimensions = [dimension_index for dimension_index in range(len(inputs.get_shape()))][1:]
        if aggregation == 'mean':
            return super().build_model(tf.reduce_mean(inputs, dimensions))


def load_stored_parameters(parameters):
    if type(parameters) is np.ndarray:
        return parameters

    if isinstance(parameters, str):
        if parameters.split('.')[-1] == 'npy':
            return np.load(parameters)
    raise NotImplementedError()


class HyperplaneProjection(PredictiveModel):

    def __init__(self, normal, bias=None, **kwargs):
        self.normal = tf.constant(load_stored_parameters(normal))
        self.bias = tf.constant(load_stored_parameters(bias)) if bias is not None \
            else tf.zeros(self.normal.get_shape().dims[0].value)
        super().__init__(**kwargs)

    def build_model(self, inputs, **kwargs):
        return super().build_model(tf.matmul(inputs, tf.transpose(self.normal)) + self.bias, **kwargs)


class ScikitLearnEstimatorTransform(PredictiveModel):

    def __init__(self, estimator, **kwargs):
        from sklearn.externals import joblib
        self.estimator = joblib.load(estimator) if isinstance(estimator, str) else estimator
        super(ScikitLearnEstimatorTransform, self).__init__(**kwargs)

    def build_model(self, inputs, **kwargs):
        return super(ScikitLearnEstimatorTransform, self).build_model(self.estimator.transform(inputs), **kwargs)


class SigmoidActivation(PredictiveModel):

    def build_model(self, inputs, **kwargs):
        return super(SigmoidActivation, self).build_model(tf.nn.sigmoid(inputs))


class SoftmaxActivation(PredictiveModel):

    def build_model(self, inputs, **kwargs):
        return super().build_model(tf.nn.softmax(inputs))


class DeepPredictiveModel(TrainableModel):
    '''A model that concatenates several PredictiveModels.'''

    def __init__(self, inner_models=None, **kwargs):
        inner_models = [(model[0], {**kwargs, **model[1]}) if isinstance(model, tuple) and len(model) > 1
                        else (model, kwargs)
                        for model in inner_models]
        super().__init__(inner_models=inner_models, **kwargs)

    def build_model(self, inputs, inner_models, **kwargs):
        current_input = inputs
        current_input_dimension = inputs.get_shape().dims[-1]
        self.inner_models_names = []
        # Instantiate every model using the provided arguments.
        for inner_model_index, (inner_model_class, inner_model_arguments) in enumerate(inner_models):
            inner_model_arguments['num_features'] = current_input_dimension
            inner_model = inner_model_class(is_inner_model=True, **inner_model_arguments)
            inner_model_name = inner_model.name + '_' + str(inner_model_index)
            with tf.variable_scope(inner_model_name):
                current_input = inner_model.build_model(current_input, **inner_model_arguments)
                current_input_dimension = [int(dimension) for dimension in current_input.get_shape().dims[1:]]

            self.inner_models_names.append(inner_model_name)

        return super().build_model(current_input, **{**kwargs, 'num_units': current_input_dimension})

    @property
    def parameters(self):
        parameters = {}
        for inner_model_name in self.inner_models_names:
            parameters.update(get_model_parameters(inner_model_name, self._session))
        return parameters


class TransformationPipeline(DeepPredictiveModel):

    def build_loss(self, label, **kwargs):
        return tf.constant(0.)


class PredictiveSequenceModel(ClassificationModel):
    '''A model that makes inferences over a sequenced input.'''

    def build_model_evolution(self, input, **kwargs):
        pass

    def build_model(self, input_sequence, num_units, **kwargs):
        '''
        Executes the model inference over each step of the inputs, evolving the model's internal representation,
        and outputs the final representation as result.
        The input_sequence is assumed to have the following shape:
            [batch_size, num_steps, num_features].
        '''
        batch_size = tf.shape(input_sequence)[0]
        self.reset(**{**kwargs, 'batch_size': batch_size})

        def loop_condition(input_sequence, output, step):
            return tf.less(step, tf.shape(input_sequence)[1])

        def loop_body(input_sequence, output, step):
            output = self.build_model_evolution(input_sequence[:, step, :], **kwargs)
            # Avoid instantiating all the reusable variables again.
            tf.get_variable_scope().reuse_variables()
            return input_sequence, output, step + 1

        loop_variables = [input_sequence, tf.zeros([batch_size, self.output_size]), tf.constant(0)]

        input_sequence, output, step = tf.while_loop(loop_condition, loop_body, loop_variables)

        return output


class DeepPredictiveSequenceModel(PredictiveSequenceModel):
    '''A model that concatenates several PredictiveSequenceModels a every timestep of the input sequence.'''

    def reset(self, inner_sequence_models, **kwargs):
        for inner_sequence_model, (_, inner_sequence_model_arguments) in zip(self._inner_sequence_models,
                                                                             inner_sequence_models):
            inner_sequence_model.reset(**{**inner_sequence_model_arguments, **kwargs})

    def build_model(self, input_sequence, inner_sequence_models, **kwargs):
        inner_sequence_models = [(model[0], {**model[1], **kwargs}) if isinstance(model, tuple) and len(model) > 1
                                 else (model, kwargs) for model in inner_sequence_models]

        self.inner_models_names = []
        self._inner_sequence_models = []
        # Instantiate every model in order to use its methods without carrying its class.
        for inner_model_index, (inner_sequence_model_class,
                                inner_sequence_model_arguments) in enumerate(inner_sequence_models):
            inner_model = inner_sequence_model_class(is_inner_model=True, **inner_sequence_model_arguments)
            self._inner_sequence_models.append(inner_model)
            self.inner_models_names.append(inner_model.name)

        return super().build_model(input_sequence, inner_sequence_models=inner_sequence_models,
                                   **{**kwargs, 'num_units': inner_model.output_size})

    def build_model_evolution(self, current_step_input, inner_sequence_models, **kwargs):
        current_input = current_step_input
        iterator = enumerate(zip(self._inner_sequence_models, inner_sequence_models))
        for inner_sequence_model_index, (inner_sequence_model_instance, (_,inner_sequence_model_arguments)) in iterator:
            with tf.variable_scope(inner_sequence_model_instance.name + '_' + str(inner_sequence_model_index)):
                current_input = inner_sequence_model_instance.build_model_evolution(current_input,
                                                                                    **inner_sequence_model_arguments)

        return current_input


class LinearNN(PredictiveModel):

    def build_model(self, inputs, num_units, **kwargs):
        return super(LinearNN, self).build_model(linear_neurons_layer(inputs, num_units, self.name))


class Dropout(PredictiveModel):

    def build_model(self, inputs, keep_prob, **kwargs):
        return tf.nn.dropout(inputs, keep_prob, name='Dropout/output')


class PredictiveRecurrentModel(PredictiveSequenceModel):

    def _build_initial_state(self, num_units, batch_size=1, **kwargs):
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
        self.state = self._build_initial_state(**kwargs)


class BasicRNN(PredictiveRecurrentModel):

    def _build_recurrent_model(self, input, state, num_units, **kwargs):
        from rnn_cell import linear
        hidden_state = tf.tanh(linear([input, state], num_units, True, scope='BasicRNN'), 'BasicRNN/hidden_state')
        return hidden_state, hidden_state


class LSTMCell(PredictiveRecurrentModel):

    def _build_initial_state(self, num_units, batch_size=1, **kwargs):
        return super()._build_initial_state(num_units * 2, batch_size, **kwargs)

    def _build_recurrent_model(self, input, state, num_units, **kwargs):
        from rnn_cell import linear
        c, h = array_ops.split(state, 2, 1)
        c = tf.identity(c, name='LSTMCell/c_state')
        h = tf.identity(c, name='LSTMCell/h_state')

        i, j, f, o = array_ops.split(linear([input, h], 4 * num_units, True), 4, 1)

        j = tf.tanh(j, name='LSTMCell/j_input')
        i = tf.sigmoid(i, name='LSTMCell/i_gate')
        f = tf.sigmoid(f, name='LSTMCell/f_gate')
        o = tf.sigmoid(o, name='LSTMCell/o_gate')
        c_ = i * j + f * c
        h_ = o * tf.tanh(c_)

        return h_, array_ops.concat([c_, h_], 1, name='LSTMCell/c_h_states')


class IterativeNN(ClassificationModel):

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
    classes_histogram = np.unique(np.array(labels).tolist(), return_counts=True)
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


def divide_set_randomly(elements, shutter_proportion, exact=False):
    if exact:
        return train_test_split(elements, test_size=shutter_proportion)
    else:
        # Here we go.
        import random

        A_fold_indices = random.sample(range(len(elements)), k=int(round(len(elements) * (1 - shutter_proportion))))
        # TODO: optimize this.
        A_fold = []
        B_fold = []

        for element_index, element in enumerate(elements):
            if element_index in A_fold_indices:
                A_fold.append(element)
            else:
                B_fold.append(element)

        return A_fold, B_fold


def unify_inputs_labels_samples(inputs, labels):
    # TODO: optimize this.
    return [sample for sample in zip(inputs, labels)]


def separate_samples_input_labels(samples):
    # TODO: optimize this.
    inputs = []
    labels = []
    for sample in samples:
        inputs.append(sample[0])
        labels.append(sample[1])

    return inputs, labels


def divide_inputs_labels_set_randomly(inputs, labels, shutter_proportion, exact=False):
    A_set, B_set = divide_set_randomly(unify_inputs_labels_samples(inputs, labels), shutter_proportion, exact)
    A_set_inputs, A_set_labels = separate_samples_input_labels(A_set)
    B_set_inputs, B_set_labels = separate_samples_input_labels(B_set)
    return A_set_inputs, B_set_inputs, A_set_labels, B_set_labels


class ArrayDataSet(DataSet):

    def __init__(self, inputs, labels=np.array([]), test_proportion=None, validation_proportion=None, labels_names=None,
                 exact_set_division=True, **kwargs):
        self.labels_names = labels_names
        self.exact_set_division = exact_set_division
        self.validation_proportion = validation_proportion
        if test_proportion is None:
            self._train_inputs, self._train_labels = inputs, labels
            self._test_inputs, self._test_labels = inputs, labels
        else:
            self._train_inputs, \
            self._test_inputs, \
            self._train_labels, \
            self._test_labels = divide_inputs_labels_set_randomly(inputs, labels, test_proportion, exact_set_division)

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
        return len(self._train_inputs)

    @property
    def num_test_samples(self):
        return len(self._train_inputs)

    @property
    def num_classes(self):
        # Consider labels as indices being counted from zero.
        return np.array(self._train_labels).max() + 1

    @property
    def num_features(self):
        sample_inputs = np.array(self.get_sample()[0])
        # If the sample is a sequence then return the length of the first step.
        return len(sample_inputs if len(sample_inputs.shape) == 1 else sample_inputs[0])

    def provide_train_validation_random_partition(self, validation_proportion=None, exact_set_division=None):
        return divide_inputs_labels_set_randomly(self._train_inputs, self._train_labels,
                                                 validation_proportion or self.validation_proportion,
                                                 exact_set_division or self.exact_set_division)

    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def test_inputs(self):
        return self._test_inputs

    @property
    def test_labels(self):
        return self._test_labels

    @property
    def inputs(self):
        return np.concatenate([self.train_inputs, self.test_inputs])

    @property
    def labels(self):
        return np.concatenate([self.train_labels, self.test_labels])

    def get_all_samples(self):
        if self._train_inputs != self._test_inputs:
            return (np.concatenate([self._train_inputs, self._test_inputs]),
                    np.concatenate([self._train_labels, self._test_labels]))
        else:
            return self._train_inputs, self._train_labels


    def get_sample(self):
        return sample_inputs_labels(*self.get_all_samples())

    def get_train_sample(self):
        return sample_inputs_labels(self._train_inputs, self._train_labels)

    def get_test_sample(self):
        return sample_inputs_labels(self._test_inputs, self._test_labels)

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


class DataFrameDataSet(ArrayDataSet):

    def __init__(self, dataset_df, sample_field=None, step_field=None, label_field=None,
                 test_proportion=None, validation_proportion=None, labels_names=None, **kwargs):

        if sample_field is not None and sample_field in dataset_df.columns and \
           label_field is not None and label_field in dataset_df.columns:
            num_samples = dataset_df[sample_field].nunique()

            labels = dataset_df[[sample_field,
                                 label_field]].drop_duplicates().set_index(sample_field)[label_field].values

            if step_field is not None and step_field in dataset_df.columns.tolist():
                # Treatment for sequential datasets.
                sequence_length_is_constant = (dataset_df[step_field].value_counts().tolist()[-1] ==
                                               dataset_df[step_field].nunique())
                if sequence_length_is_constant:
                    self.steps_per_sample = dataset_df[step_field].nunique()
                    num_features = len([column for column in dataset_df.columns if column not in [sample_field,
                                                                                                  step_field,
                                                                                                  label_field]])

                    inputs = dataset_df.drop(label_field, axis=1).set_index([sample_field, step_field]).as_matrix()\
                                .reshape([num_samples, self.steps_per_sample, num_features])
                else:
                    self.steps_per_sample = None
                    inputs = np.array([sample_sequence_df[[column
                                                           for column in sample_sequence_df.columns.tolist()
                                                           if not column in [sample_field, label_field]]].values
                                       for (sample, label), sample_sequence_df
                                       in dataset_df.groupby([sample_field, label_field])])
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

        super().__init__(inputs, labels,
                         test_proportion=test_proportion, validation_proportion=validation_proportion,
                         labels_names=labels_names, **kwargs)


class CSVDataSet(DataFrameDataSet):

    def __init__(self, path_or_buffer, **kwargs):
        import pandas as pd
        super().__init__(pd.read_csv(path_or_buffer, **kwargs), **kwargs)


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
        self.inputs_key = inputs_key
        self.labels_key = labels_key
        self.labels_names = labels_names

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

    @property
    def inputs(self):
        return data_partition(self.data_sequences_sets, self.inputs_key)

    @property
    def labels(self):
        return data_partition(self.data_sequences_sets, self.labels_key)

    def provide_train_validation_random_partition(self, validation_proportion=None):
        validation_proportion = validation_proportion or self.validation_proportion
        if self.exact_merge:
            return super(SequentialDataMerge, self).provide_train_validation_random_partition(validation_proportion)
        else:
            train_sequence_sets = self.data_sequences_sets[:self.num_train_sequence_sets]

            train_fold, validation_fold = divide_set_randomly(train_sequence_sets, validation_proportion)

            train_fold_inputs = SequentialDataMerge([data_sequence_set[:, self.inputs_key]
                                                    for data_sequence_set in train_fold],
                                                    test_proportion=0, validation_proportion=0,
                                                    labels_names=self.labels_names, exact_merge=False,
                                                    inputs_key=self.inputs_key, labels_key=self.labels_key)
            train_fold_labels = SequentialDataMerge([data_sequence_set[:, self.labels_key]
                                                     for data_sequence_set in train_fold],
                                                    test_proportion=1, validation_proportion=0,
                                                    labels_names=self.labels_names, exact_merge=False,
                                                    inputs_key=self.inputs_key, labels_key=self.labels_key)

            validation_fold_inputs = SequentialDataMerge([data_sequence_set[:, self.inputs_key]
                                                          for data_sequence_set in validation_fold],
                                                         test_proportion=0, validation_proportion=1,
                                                         labels_names=self.labels_names, exact_merge=False,
                                                         inputs_key=self.inputs_key, labels_key=self.labels_key)
            validation_fold_labels = SequentialDataMerge([data_sequence_set[:, self.labels_key]
                                                          for data_sequence_set in validation_fold],
                                                         test_proportion=1, validation_proportion=1,
                                                         labels_names=self.labels_names, exact_merge=False,
                                                         inputs_key=self.inputs_key, labels_key=self.labels_key)

            return train_fold_inputs, validation_fold_inputs, train_fold_labels, validation_fold_labels

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
        return self.data_sequences_sets[0].shape[-1]

    @property
    def steps_per_sample(self):
        return self.data_sequences_sets[0].shape[-2]

    @property
    def num_classes(self):
        return max(*([data_sequences_set.num_classes for data_sequences_set in self.data_sequences_sets] + [0]))

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

    def __init__(self, files_folder_path, epoch_duration, low_frequencies_cut=None, high_frequencies_cut=None,
                 transformation=None, **kwargs):
        self.files_folder_path = files_folder_path
        self.epoch_duration = epoch_duration
        self.low_frequencies_cut = low_frequencies_cut
        self.high_frequencies_cut = high_frequencies_cut

        # If there is a next onset(to compare with) and it's steps_per_sample ahead, then is a valid epoch.
        signal_classification = self.signal_classification
        steps_per_sample = self.steps_per_sample
        self.valid_samples = [(onset, label) for onset_index, (onset, label) in enumerate(signal_classification)
                              if onset_index + 1 < len(signal_classification)
                              and (signal_classification[onset_index + 1][0] - onset) == steps_per_sample]
        self.transformation = transformation if not isinstance(transformation, str) \
                                             else self._build_transformation(transformation)

    def _build_transformation(self, transformation):
        if transformation == 'standardization':
            from sklearn.preprocessing import StandardScaler
            transformation = StandardScaler()
            transformation.partial_fit(self.eeg_signals.to_data_frame().values[:, :self.num_features])
            return transformation
        raise NotImplementedError()

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

    @property
    def inputs_key(self):
        return self.eeg_signal_sample_position

    @property
    def labels_key(self):
        return self.label_sample_position

    def _transform_raw_signal_data(self, raw_data):
        import mne
        # Frequencies filtering.
        result = mne.filter.filter_data(raw_data, self.eeg_signals.info.get('sfreq'),
                                        self.low_frequencies_cut, self.high_frequencies_cut).T

        # Feature transformation(standardization or others).
        return self.transformation.transform(result) if self.transformation is not None else result

    def __getitem__(self, key):
        if np.isscalar(key):
            onset, label = self.valid_samples[key]
            # Omit empty stim channel and time dimmension.
            return self._transform_raw_signal_data(self.eeg_signals[:-1, onset:onset + self.steps_per_sample][0]), label
        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                if key[1] == self.labels_key:
                    return np.array(self.valid_samples)[key]
                if key[1] == self.inputs_key:
                    return KeyPartialSequentialData(data_key=self.inputs_key, complete_sequential_data=self)


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

    def __init__(self, experiments_data_folders, epoch_duration, **kwargs):
        eeg_experiments_data = [FmedLfaEegExperimentData(experiments_data_folder, epoch_duration, **kwargs)
                                for experiments_data_folder
                                in experiments_data_folders]
        super().__init__(eeg_experiments_data, inputs_key=eeg_experiments_data[0].inputs_key,
                         labels_key=eeg_experiments_data[0].labels_key, **kwargs)