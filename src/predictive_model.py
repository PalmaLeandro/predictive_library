import logging
import math
import numpy as np
import tensorflow as tf

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
    plt.xticks(rotation=90)
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
        return samples_indices_batches + [samples_indices[num_batches * batch_size:]]  # Batches' leftover.


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
            #super().__init__(cluster_machines)
            self._model_persistence_dir = model_persistence_dir
            self._erase_model_persistence_dir = erase_model_persistence_dir

            # ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(cluster_machines),
            #                                                               tf.contrib.training.byte_size_load_fn)
            if True:
            #with tf.device(tf.train.replica_device_setter(self._cluster, ps_strategy=ps_strategy)):
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

            self._session = tf.Session()
            self._task_index = 0
            # tf.train.MonitoredTrainingSession(master=self._worker_server.target,
                           #                                   is_chief=(self._task_index == 0),
                           #                                   checkpoint_dir=self._model_persistence_dir,
                           #                                   config=self._cluster_config)
            self._session.run(self._global_initializer)

        self.output_size = num_units

    def build_model(self, inputs, name=None, **kwargs):
        '''
        Defines the model output to be used to infer a prediction, given the provided input.
        Needs to be implemented by subclasses.
        '''
        output = tf.identity(inputs, name=name or self.name)
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
                                                            data_set.labels_names)
        return confusion_matrix_df['Accuracy'].sum() / len(confusion_matrix_df['Accuracy'])


class TrainableModel(PredictiveModel):

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


class BatchNormalization(PredictiveModel):

    def build_model(self, inputs, dimensions=None, **kwargs):
        if dimensions is None:
            # not_batch_dimensions_indexes:
            dimensions = [dimension_index for dimension_index in range(len(inputs.get_shape()))][1:]
        max = tf.reduce_max(inputs, dimensions)
        min = tf.reduce_min(inputs, dimensions)
        var = max - min
        dimensions_mask = [1 if dimension_index not in dimensions else inputs.get_shape().dims[dimension_index].value
                           for dimension_index in range(len(inputs.get_shape()))]
        min = tf.tile(tf.expand_dims(min, dimensions), dimensions_mask)
        var = tf.tile(tf.expand_dims(var, dimensions), dimensions_mask)
        return super().build_model((inputs - min) / var)


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


class TransformationPipeline(PredictiveModel):
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


class DeepPredictiveModel(TrainableModel, ClassificationModel, TransformationPipeline):
    ''''''


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


class MovingAverageSubtraction(PredictiveModel):

    def build_model(self, inputs, num_features, num_neighbouring_bins=2, **kwargs):
        num_pair_bins_to_average = num_neighbouring_bins // 2

        diagonal = tf.eye(num_features[-1] + num_pair_bins_to_average * 2)

        cumulative_diagonal = tf.identity(diagonal, name='diagonal_matrix')
        for neighbouring_bin_pair_index in range(1, num_pair_bins_to_average):
            cumulative_diagonal += tf.manip.roll(diagonal, neighbouring_bin_pair_index, 0)
            cumulative_diagonal += tf.manip.roll(diagonal, -neighbouring_bin_pair_index, 0)

        cumulative_diagonal = cumulative_diagonal[num_pair_bins_to_average:num_features[-1] + num_pair_bins_to_average,
                                                  num_pair_bins_to_average:num_features[-1] + num_pair_bins_to_average]
        cumulative_diagonal /= (num_pair_bins_to_average * 2 + 1)

        bins_average = tf.tensordot(inputs, cumulative_diagonal, [[-1], [0]])

        return super().build_model(inputs - bins_average)