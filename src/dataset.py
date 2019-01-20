import logging
import random
import numpy as np

from sklearn.model_selection import train_test_split


def dataset_class_distribution(labels, labels_names=None, show_table=True, plot_chart=True, ax=None, **kwargs):
    classes_histogram = np.unique(np.array(labels).tolist(), return_counts=True)
    if not show_table and not plot_chart:
        return classes_histogram
    import pandas as pd
    labels_names = labels_names or [str(class_index) for class_index in classes_histogram[0]]
    classes_histogram_df = pd.DataFrame({'label': classes_histogram[0],
                                         'class': [labels_names[class_index]
                                                   for class_index, class_num_samples
                                                   in zip(*classes_histogram)],
                                         'num_samples': [class_num_samples
                                                         for class_index, class_num_samples
                                                         in zip(*classes_histogram)]}).sort_values('label')
    if plot_chart:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), sharex=True, sharey=True, **kwargs)
            plt.xticks(rotation=90)

        ax.bar(classes_histogram_df['class'], classes_histogram_df['num_samples'])

    if show_table:
        classes_histogram_df.set_index(['label', 'class'], inplace=True)

    return classes_histogram_df, ax


def sample_input_label(inputs, labels=None, return_sample_index=False):
    sample_index = random.randrange(len(inputs))
    if labels is not None:
        assert len(labels) == len(inputs), 'Inputs an labels lengths don\'t match.'

        return ((inputs[sample_index], labels[sample_index], sample_index) if return_sample_index is True
                else (inputs[sample_index], labels[sample_index]))
    else:
        return inputs[sample_index], sample_index if return_sample_index is True else inputs[sample_index]


def divide_set_randomly(elements, shutter_proportion, exact=False):
    if exact:
        return train_test_split(elements, test_size=shutter_proportion)
    else:
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


def separate_samples_input_labels(samples, inputs_key=0, labels_key=1):
    # TODO: optimize this.
    inputs = []
    labels = []
    for sample in samples:
        inputs.append(sample[0])
        labels.append(sample[1])

    return inputs, labels


def divide_inputs_labels_set_randomly(inputs, labels, shutter_proportion, exact=False, inputs_key=0, labels_key=1):
    A_set, B_set = divide_set_randomly(unify_inputs_labels_samples(inputs, labels), shutter_proportion, exact)
    A_set_inputs, A_set_labels = separate_samples_input_labels(A_set, inputs_key=0, labels_key=1)
    B_set_inputs, B_set_labels = separate_samples_input_labels(B_set, inputs_key=0, labels_key=1)
    return A_set_inputs, B_set_inputs, A_set_labels, B_set_labels


def random_choice(probability_of_success):
    return random.random() < probability_of_success


def bucket_indices(indices, cuts):
    buckets = [[] for bucket in range(len(cuts) + 1)]
    for index in indices:
        buckets[min([cut_index for cut_index, cut in enumerate(cuts) if index < cut])].append(index)
    return buckets


class DataSet(object):

    def __init__(self, validation_proportion=None, labels_names=None, features_names=None, exact=True,
                 inputs_key=0, labels_key=1, **kwargs):
        self.validation_proportion = validation_proportion
        self._labels_names = labels_names
        self._features_names = features_names
        self.exact = exact
        self.inputs_key = inputs_key
        self.labels_key = labels_key

    @property
    def inputs(self):
        pass

    @property
    def labels(self):
        pass

    @property
    def train_inputs(self):
        pass

    @property
    def train_labels(self):
        pass

    @property
    def test_inputs(self):
        pass

    @property
    def test_labels(self):
        pass

    def get_all_samples(self):
        pass

    def get_sample_batch(self, batch_size):
        pass

    @property
    def labels_names(self):
        return self._labels_names if self._labels_names is not None \
               else [str(label) for label in range(self.num_classes)]

    @property
    def features_names(self):
        return self._features_names if self._features_names is not None \
               else [str(feature) for feature in range(self.num_features)]

    @property
    def num_samples(self):
        return len(self.inputs)

    def __len__(self):
        return self.num_samples

    @property
    def num_train_samples(self):
        return len(self.train_inputs)

    @property
    def num_test_samples(self):
        return len(self.test_inputs)

    @property
    def test_proportion(self):
        return self.num_test_samples / float(self.num_test_samples + self.num_train_samples)

    @property
    def num_classes(self):
        # Consider labels as indices being counted from zero.
        return np.array(self.train_labels).max() + 1

    @property
    def num_features(self):
        sample_inputs = np.array(self.get_sample()[0])
        # If the sample is a sequence then return the length of the first step.
        return len(sample_inputs if len(sample_inputs.shape) == 1 else sample_inputs[0])

    def provide_train_validation_partition(self, validation_proportion=None, exact=None):
        return divide_inputs_labels_set_randomly(self.train_inputs, self.train_labels,
                                                 validation_proportion or self.validation_proportion,
                                                 exact or self.exact)

    def get_sample(self, return_sample_index=False, return_sample_input=True):
        return sample_input_label(self.inputs if return_sample_input else self.labels,
                                  self.labels if return_sample_input else None,
                                  return_sample_index=return_sample_index)

    def get_train_sample(self, return_sample_index=False, return_sample_input=True):
        return sample_input_label(self.train_inputs if return_sample_input else self.train_labels,
                                  self.train_labels if return_sample_input else None,
                                  return_sample_index=return_sample_index)

    def get_test_sample(self, return_sample_index=False, return_sample_input=True):
        return sample_input_label(self.test_inputs if return_sample_input else self.test_labels,
                                  self.test_labels if return_sample_input else None,
                                  return_sample_index=return_sample_index)

    def classes_distribution(self, ax=None):
        classes_histogram_df, _ax = dataset_class_distribution(self.labels, self.labels_names, ax=ax)
        if ax is None:
            _ax.set_title('Dataset classes distribution')
            import matplotlib.pyplot as plt
            plt.show()

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
        return pd.pivot_table(pd.concat([train_samples_histogram_df, test_samples_histogram_df]),
                              index=['label', 'class'],
                              columns='dataset')

    def balance_classes(self, classes_to_balance=None, reference_class=None):
        classes_to_balance = list(range(self.num_classes)) if classes_to_balance is None else classes_to_balance
        classes_to_balance = [self._labels_names.index(class_label) if isinstance(class_label, str) else class_label
                              for class_label in classes_to_balance]

        classes_labels, classes_num_samples = np.unique(np.array(self.labels).tolist(), return_counts=True)
        classes_num_samples = [classes_num_samples[classes_labels.tolist().index(class_label)]
                               if class_label in classes_labels else 0
                               for class_label in range(self.num_classes)]

        classes_to_balance_num_samples = [class_num_samples
                                          for class_label, class_num_samples
                                          in zip(classes_labels, classes_num_samples)
                                          if class_label in classes_to_balance]
        reference_class = np.argmin(classes_to_balance_num_samples) if reference_class is None else reference_class
        reference_class = self._labels_names.index(reference_class) if isinstance(reference_class, str) \
                          else reference_class

        num_reference_class_samples = classes_num_samples[classes_to_balance[reference_class]]

        classes_num_samples_to_vary = [num_reference_class_samples - class_num_samples
                                       if class_label in classes_to_balance else 0
                                       for class_label, class_num_samples
                                       in enumerate(classes_num_samples)]

        self.modify_classes_distribution(classes_num_samples_to_vary)

    def modify_classes_distribution(self, classes_num_samples_to_vary):
        self.remove_samples(np.concatenate(self.samples_by_labels([(class_label, abs(frequency_change_requested))
                                                                   for class_label, frequency_change_requested
                                                                   in enumerate(classes_num_samples_to_vary)
                                                                   if frequency_change_requested < 0],
                                                                  return_samples_indices=True,
                                                                  return_samples_inputs=False)))

        samples_to_repeat_indices = self.samples_by_labels([(class_label, abs(frequency_change_requested))
                                                    for class_label, frequency_change_requested
                                                    in enumerate(classes_num_samples_to_vary)
                                                    if frequency_change_requested > 0],
                                                    return_samples_indices=True,
                                                    return_samples_inputs=False)
        self.add_samples(DataSubSet(self, inputs_key=self.inputs_key, data_indices=samples_to_repeat_indices),
                         DataSubSet(self, inputs_key=self.labels_key, data_indices=samples_to_repeat_indices))

    def blend_classes(self, classes_to_blend, blend_name=None):
        classes_to_blend = [self.labels_names.index(class_label) if isinstance(class_label, str) else class_label
                            for class_label in classes_to_blend]
        blend_label = min(classes_to_blend)
        mapping = {class_to_blend: blend_label for class_to_blend in classes_to_blend if blend_label != class_to_blend}
        for class_label in range(min(classes_to_blend), self.num_classes):
            if class_label not in classes_to_blend:
                lower_classes = [class_to_blend for class_to_blend in classes_to_blend if class_to_blend < class_label]
                class_shift = len(lower_classes) - 1
                if class_shift > 0:
                    mapping.update({class_label: class_label - class_shift})

        self._replace_labels(mapping)

        blend_name = blend_name if blend_name is not None \
            else ' + '.join([self.labels_names[class_label] for class_label in classes_to_blend])

        self.replace_labels_names(self.labels_names[:blend_label] + [blend_name] +
                                  [class_name for class_label, class_name in enumerate(self.labels_names)
                                   if class_label in mapping.keys() and class_label not in classes_to_blend])

    def sample_at(self, key):
        return self.inputs[key], self.labels[key]

    def __getitem__(self, key):
        if np.isscalar(key):
            return self.sample_at(key)
        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                if key[1] == self.labels_key:
                    return self.labels[key[0]]
                if key[1] == self.inputs_key:
                    return self.inputs[key[0]]

    def add_samples(self, inputs_to_add, labels_to_add, **kwargs):
        pass

    def remove_samples(self, samples_indices):
        pass

    def replace_labels(self, class_label_mapping, prevent_former_labels_collision=True):
        mapping = {(self.labels_names.index(former_label) if isinstance(former_label, str) else former_label): new_label
                   for former_label, new_label in class_label_mapping.items()}

        if prevent_former_labels_collision:
            former_labels_mapping = {}
            for former_label in class_label_mapping.values():
                if former_label not in former_labels_mapping:
                    references = [new_label for new_label, old_label in mapping.items() if old_label == former_label]
                    former_labels_mapping.update({former_label: min(references)})

            mapping.update(former_labels_mapping)

        self._replace_labels(mapping)

    def _replace_labels(self, mapping, samples_indices=None):
        pass

    def replace_labels_names(self, new_labels_names):
        self._labels_names = new_labels_names

    def samples_by_labels(self, classes_samples_requested, return_samples_indices=False, return_samples_inputs=True):
        ''' class_samples_requested is expected to be a list of tuples or strings like
            [(class1, class1_required_samples), class2, ...]'''
        classes_samples_requested = [class_samples_requested if isinstance(class_samples_requested, (list, tuple))
                                     else (class_samples_requested, None)
                                     for class_samples_requested in classes_samples_requested]

        classes_samples_requested = [[self.labels_names.index(class_label) if isinstance(class_label, str)
                                      else class_label, requested_samples]
                                     for class_label, requested_samples in classes_samples_requested]
        classes_to_sample_labels = [class_label for class_label, requested_samples in classes_samples_requested]

        samples_indices_by_labels = [[] for class_label in range(len(classes_samples_requested))]

        shuffled_sample_indices = list(range(self.num_samples))
        import random
        random.shuffle(shuffled_sample_indices)
        shuffled_sample_index = 0
        while (shuffled_sample_index < len(shuffled_sample_indices)
          and any([requested_samples != 0 for class_label, requested_samples in classes_samples_requested])):

            true_sample_index = shuffled_sample_indices[shuffled_sample_index]
            sample_label = self.labels[true_sample_index]
            shuffled_sample_index += 1
            if sample_label in classes_to_sample_labels:
                samples_indices_by_labels[classes_to_sample_labels.index(sample_label)].append(true_sample_index)

                sample_class_label, sample_class_requested_samples = \
                    classes_samples_requested[classes_to_sample_labels.index(sample_label)]

                if sample_class_requested_samples is not None:
                    if sample_class_requested_samples > 0:
                        classes_samples_requested[classes_to_sample_labels.index(sample_label)][1] -= 1
                    elif sample_class_requested_samples < 0:
                        classes_samples_requested[classes_to_sample_labels.index(sample_label)][1] += 1

        if return_samples_inputs:
            samples_by_label = [[[],[]] * len(classes_samples_requested)]
            for label_index, label in enumerate(classes_samples_requested):
                class_samples_inputs = np.array(self[samples_indices_by_labels[label_index]])[:, 0]
                class_samples_labels = np.repeat(label, len(class_samples_inputs))
                samples_by_label[label_index] = class_samples_inputs, class_samples_labels

            return (samples_by_label, samples_indices_by_labels) if return_samples_indices else samples_by_label
        else:
            return samples_indices_by_labels


class ArrayDataSet(DataSet):

    def __init__(self, inputs, labels=np.array([]), test_proportion=None, **kwargs):
        super().__init__(**kwargs)
        if test_proportion is None:
            self._train_inputs, self._train_labels = inputs, labels
            self._test_inputs, self._test_labels = inputs, labels
        else:
            self._train_inputs, \
            self._test_inputs, \
            self._train_labels, \
            self._test_labels = divide_inputs_labels_set_randomly(inputs, labels, test_proportion, self.exact,
                                                                  self.inputs_key, self.labels_key)

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
        if self._train_inputs is self._test_inputs:
            return self._train_inputs
        else:
            return np.concatenate([self.train_inputs, self.test_inputs])

    @property
    def labels(self):
        if self._train_labels is self._test_labels:
            return self._train_labels
        else:
            return np.concatenate([self._train_labels, self._test_labels])

    def add_samples(self, inputs_to_add, labels_to_add, **kwargs):
        samples_division = divide_inputs_labels_set_randomly(inputs_to_add, labels_to_add, self.test_proportion, False)
        train_inputs_to_add, test_inputs_to_add, train_labels_to_add, test_labels_to_add = samples_division

        self._train_inputs = np.concatenate([self.train_inputs, train_inputs_to_add])
        self._train_labels = np.concatenate([self.train_labels, train_labels_to_add])
        self._test_inputs = np.concatenate([self.test_inputs, test_inputs_to_add])
        self._test_labels = np.concatenate([self.test_labels, test_labels_to_add])

    def remove_samples(self, samples_indices):
        train_samples, test_samples = bucket_indices(samples_indices, [self.num_train_samples, self.num_samples])
        test_samples -= self.num_train_samples
        self._train_inputs = np.delete(self._train_inputs, train_samples, 0)
        self._train_labels = np.delete(self._train_labels, train_samples, 0)
        self._test_inputs = np.delete(self._test_inputs, test_samples, 0)
        self._test_labels = np.delete(self._test_labels, test_samples, 0)

    def _replace_labels(self, mapping, samples_indices=None):
        if samples_indices is None:
            self._train_labels = np.array([mapping.get(label, label) for label in self._train_labels])
            self._test_labels = np.array([mapping.get(label, label) for label in self._test_labels])
        else:
            train_indices_to_affect = []
            test_indices_to_affect = []
            for sample_index in samples_indices:
                if sample_index < self.num_train_samples:
                    train_indices_to_affect.append(sample_index)
                else:
                    test_indices_to_affect.append(sample_index)
                    
            self._train_labels = np.array([mapping.get(label, label) if index in train_indices_to_affect else label
                                           for index, label in enumerate(self._train_labels)])
            self._test_labels = np.array([mapping.get(label, label) if index in test_indices_to_affect else label
                                           for index, label in enumerate(self._test_labels)])


class DataFrameDataSet(DataSet):

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

                    inputs = dataset_df.drop(label_field, axis=1).set_index([sample_field, step_field]).as_matrix() \
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


class SequentialDataSet(DataSet):

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
    def units_per_sample(self):
        return 1

    @property
    def shape(self):
        return [self.num_samples, self.steps_per_sample, self.num_features]

    def plot_sample(self, sample=None, sample_index=None, axes=None, step_pace=None, first_ax_title=None):
        import matplotlib.pyplot as plt

        _axes = axes if axes is not None else plt.subplots(self.num_features, figsize=(16, 3 * self.num_features))[1]

        if sample is not None:
            sequence_inputs, sequence_label = sample
        else:
            if sample_index is not None:
                sequence_inputs, sequence_label = self.sample_at(sample_index)
            else:
                sequence_inputs, sequence_label, sample_index = self.get_sample(return_sample_index=True)

        steps_per_sample, num_features = sequence_inputs.shape

        first_ax_title = ((first_ax_title + '\n') if first_ax_title is not None else '')
        first_ax_title += 'Sample {sample_index}, Class {sample_class}'
        sample_class = sequence_label if isinstance(sequence_label, str) else self.labels_names[sequence_label]
        _axes[0].set_title(first_ax_title.format(sample_index=sample_index, sample_class=sample_class))

        step_pace = step_pace if step_pace is not None else self.units_per_sample / float(self.steps_per_sample)
        for dimension_index in range(num_features):
            _axes[dimension_index].plot(np.array(range(steps_per_sample)) * step_pace,
                                        sequence_inputs[:, dimension_index], label=self.features_names[dimension_index])
            _axes[dimension_index].legend()

        if axes is None:
            plt.show()


class DataSubSet(DataSet):
    """Class that wraps a DataSet in order to retrieve only part of it, reusing its main functionality.
       This is particularly useful to retrieve inputs or labels separately in order to build an ArrayDataSet."""

    def __init__(self, complete_data_set, data_key=None, data_indices=None, **kwargs):
        self.data_key = data_key
        self.data_indices = data_indices if data_indices is not None else list(range(len(complete_data_set)))
        self.complete_data_set = complete_data_set
        super().__init__(**kwargs)

    def __getitem__(self, key):
        full_data = self.complete_data_set[self.data_indices[key]]
        return full_data[self.data_key] if self.data_key is not None else full_data

    def tolist(self):
        return self.complete_data_set[self.data_indices, self.data_key]

    @property
    def inputs(self):
        return self.complete_data_set.inputs[self.data_indices]

    @property
    def labels(self):
        return self.complete_data_set.labels[self.data_indices]

    @property
    def num_samples(self):
        return len(self.data_indices)

    @property
    def num_features(self):
        return self.complete_data_set.num_features

    @property
    def num_classes(self):
        return self.complete_data_set.num_classes

    @property
    def labels_names(self):
        return self._labels_names if self._labels_names is not None else self.complete_data_set.labels_names

    @property
    def features_names(self):
        return self._features_names if self._features_names is not None else self.complete_data_set.features_names

    def _replace_labels(self, mapping, samples_indices=None):
        samples_indices = self.data_indices if samples_indices is None else list(set(samples_indices) &
                                                                                 set(self.data_key))
        self.complete_data_set._replace_labels(mapping, samples_indices)


class SequentialDataSubSet(DataSubSet, SequentialDataSet):
    """Class that wraps a SequentialDataSet in order to retrieve only part of it, reusing its main functionality.
       This is particularly useful to retrieve inputs or labels separately to build an ArrayDataSet."""

    @property
    def steps_per_sample(self):
        return self.complete_data_set.steps_per_sample


def coarse_subsets_partition(subsets, ratio, shuffle=True):
    if ratio not in (0, 1, None):
        subsets = np.array(subsets)
        if shuffle:
            np.random.shuffle(subsets)
        subsets_cumulative_lengths = np.cumsum([len(subset) for subset in subsets])
        total_length = subsets_cumulative_lengths[-1]
        cut_index = min(*([index for index, cumulative_length in enumerate(subsets_cumulative_lengths)
                           if cumulative_length > (total_length * ratio)] + [len(subsets_cumulative_lengths) - 1]))
        A = subsets[:cut_index]
        B = subsets[cut_index:]
    else:
        A = subsets
        B = []

    return A, B


class DataSetsMerge(DataSet):

    def __init__(self, data_sets, test_proportion=0., shuffle=False, **kwargs):
        super().__init__(**kwargs)
        data_sets_samples_indices = np.expand_dims(np.array(range(sum([len(data_set) for data_set in data_sets]))), 0).T
        train_samples_indices, test_samples_indices = coarse_subsets_partition(data_sets_samples_indices,
                                                                               1 - test_proportion,
                                                                               shuffle=shuffle)
        train_samples_indices = np.array(train_samples_indices).flatten()
        test_samples_indices = np.array(test_samples_indices).flatten()
        self.data_sets = data_sets
        self._train_inputs = DataSubSet(self, self.inputs_key, train_samples_indices)
        self._train_labels = DataSubSet(self, self.labels_key, train_samples_indices)
        self._test_labels = DataSubSet(self, self.labels_key, test_samples_indices)
        self._test_inputs = DataSubSet(self, self.inputs_key, test_samples_indices)

    @property
    def labels_names(self):
        return self._labels_names if self._labels_names is not None else self.data_sets[0].labels_names

    @property
    def features_names(self):
        return self._features_names if self._features_names is not None else self.data_sets[0].features_names

    @property
    def inputs(self):
        return DataSubSet(self, self.inputs_key)

    @property
    def labels(self):
        return np.concatenate([data_set.labels for data_set in self.data_sets])

    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def train_labels(self):
        labels = self.labels
        return np.array([labels[index] for index in self._train_labels.data_indices])

    @property
    def test_inputs(self):
        return self._test_inputs

    @property
    def test_labels(self):
        labels = self.labels
        return np.array([labels[index] for index in self._test_labels.data_indices])

    def provide_train_validation_partition(self, k_fold=None, validation_proportion=None):
        '''For training and validating by crossvalidation which requires to define the requested k_fold
        or by bootstraping(sampling with repetitions).'''
        validation_proportion = validation_proportion or self.validation_proportion

        train_samples_indices = self.train_inputs.data_indices
        if k_fold is None:
            k_fold = k_fold % (1 // validation_proportion)
            validation_fold_length = validation_proportion * len(train_samples_indices)
            validation_fold_offset = validation_proportion * len(train_samples_indices) * k_fold
            validation_samples_indices_fold = \
                train_samples_indices[validation_fold_offset:validation_fold_offset + validation_fold_length]
            train_samples_indices_fold = [index for index in train_samples_indices
                                          if index not in validation_samples_indices_fold]
        else:
            train_samples_indices_fold, validation_samples_indices_fold = divide_set_randomly(train_samples_indices,
                                                                                              validation_proportion)

        return DataSubSet(self, data_key=self.inputs_key, data_indices=train_samples_indices_fold), \
               DataSubSet(self, data_key=self.inputs_key, data_indices=validation_samples_indices_fold), \
               DataSubSet(self, data_key=self.labels_key, data_indices=train_samples_indices_fold), \
               DataSubSet(self, data_key=self.labels_key, data_indices=validation_samples_indices_fold)

    @property
    def num_samples(self):
        return sum([len(data_set) for data_set in self.data_sets])

    @property
    def num_train_samples(self):
        return len(self._train_inputs)

    @property
    def num_test_samples(self):
        return len(self._test_inputs)

    @property
    def num_features(self):
        return self.data_sets[0].shape[-1]

    @property
    def num_classes(self):
        return max(*([data_set.num_classes for data_set in self.data_sets] + [0]))

    def sample_at(self, index):
        if index < self.num_samples:
            datasets_limits_cumsum = np.cumsum([len(data_set)
                                                for data_set
                                                in self.data_sets]).tolist()
            data_set_index, sample_index = [(data_set_index, index - dataset_start_index)
                                            for data_set_index, (dataset_start_index,
                                                                 dataset_end_index)
                                            in enumerate(zip([0] + datasets_limits_cumsum,
                                                             datasets_limits_cumsum + [0]))
                                            if index < dataset_end_index][0]
            return self.data_sets[data_set_index][sample_index]
        else:
            raise IndexError('Requested sample at {} is out of bounds.'.format(index))

    def get_sample(self, return_sample_index=False):
        import random
        sample_index = random.randint(0, self.num_samples)
        sample_inputs, sample_label = self.sample_at(sample_index)
        return (sample_inputs, sample_label, sample_index) if return_sample_index else (sample_inputs, sample_label)

    def tolist(self):
        return [self.sample_at(sample_index) for sample_index in range(self.num_samples)]

    def add_samples(self, inputs, labels, **kwargs):
        samples_division = divide_inputs_labels_set_randomly(inputs, labels, self.test_proportion)
        train_sequences_to_add, test_sequences_to_add, train_labels_to_add, test_labels_to_add = samples_division
        train_sets = self.data_sets[:self.num_train_data_sets]
        train_sets.append(ArrayDataSet(train_sequences_to_add, train_labels_to_add))
        test_sets = self.data_sets[self.num_train_data_sets:]
        test_sets.append(ArrayDataSet(test_sequences_to_add, test_labels_to_add))
        self.num_train_data_sets += 1
        self.data_sets = train_sets + test_sets

    def remove_samples(self, samples_indices):
        cumulative_data_sets_length = np.cumsum([len(data_set) for data_set in self.data_sets])
        data_sets_samples = bucket_indices(samples_indices, cumulative_data_sets_length)
        offsets = [0] + cumulative_data_sets_length[:-1].tolist()
        for data_set, data_set_offset, samples_indices_to_remove in zip(self.data_sets, offsets, data_sets_samples):
            data_set.remove_samples(np.array(samples_indices_to_remove) - data_set_offset)

    def _replace_labels(self, mapping, samples_indices=None):
        if samples_indices is None:
            for data_set in self.data_sets:
                data_set._replace_labels(mapping, samples_indices=None)
        else:
            cumulative_data_sets_length = np.cumsum([len(data_set) for data_set in self.data_sets])
            data_sets_samples = bucket_indices(samples_indices, cumulative_data_sets_length)
            offsets = [0] + cumulative_data_sets_length[:-1].tolist()
            for data_set, data_set_offset, samples_indices_to_remove in zip(self.data_sets, offsets, data_sets_samples):
                data_set._replace_labels(mapping, np.array(samples_indices_to_remove) - data_set_offset)

    def replace_labels_names(self, new_labels_names):
        super().replace_labels_names(new_labels_names)
        for data_set in self.data_sets:
            data_set.replace_labels_names(new_labels_names)


class SequentialDataSetsMerge(DataSetsMerge, SequentialDataSet):

    @property
    def steps_per_sample(self):
        return self.data_sets[0].shape[-2]

    @property
    def units_per_sample(self):
        return self.data_sets[0].units_per_sample


class EpochEegExperimentDataSet(SequentialDataSet):

    def __init__(self, files_folder_path, epoch_duration, low_frequencies_cut=None, high_frequencies_cut=None,
                 transformation=None, **kwargs):
        super().__init__(**kwargs)
        self.files_folder_path = files_folder_path
        self.epoch_duration = epoch_duration
        self.low_frequencies_cut = low_frequencies_cut
        self.high_frequencies_cut = high_frequencies_cut

        self.valid_samples = self.validate_samples(self.signal_classification)
        self.transformation = transformation if not isinstance(transformation, str) \
            else self._build_transformation(transformation)
        import mne
        self.picks = mne.pick_types(self.eeg_signals.info, eeg=True, eog=False, emg=False, stim=False)

    def validate_samples(self, signal_classification, labels_mapping={}, mapping_indices=None):
        # If there is a next onset(to compare with) and it's steps_per_sample ahead, then is a valid epoch.
        return np.array([(onset, labels_mapping.get(label, label))
                         for onset_index, (onset, label) in enumerate(signal_classification)
                         if onset_index + 1 < len(signal_classification)
                         and (signal_classification[onset_index + 1][0] - onset) == self.steps_per_sample])

    def _build_transformation(self, transformation):
        if transformation == 'standardization':
            from sklearn.preprocessing import StandardScaler
            transformation = StandardScaler()
            transformation.partial_fit(self.eeg_signals.to_data_frame().values[:, :self.num_features])
            return transformation
        raise NotImplementedError()

    @property
    def units_per_sample(self):
        return self.epoch_duration

    @property
    def eeg_signals(self):
        import mne
        set_filename = list_folder_files_with_extension(self.files_folder_path, 'set')[0]
        complete_experiment_recording = mne.io.read_raw_eeglab(input_fname=set_filename, preload=False, eog='auto')
        return complete_experiment_recording

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
        return len(self.picks)

    @property
    def features_names(self):
        # Omit stim channel
        return np.array(self.eeg_signals.ch_names)[self.picks]

    @property
    def num_classes(self):
        # Plus one since classes are considerated to be enumerated from 0.
        return np.array(self.valid_samples)[:, self.labels_key].max() + 1

    def _transform_raw_signal_data(self, raw_data):
        import mne
        # Frequencies filtering.
        result = mne.filter.filter_data(raw_data, self.eeg_signals.info.get('sfreq'),
                                        self.low_frequencies_cut, self.high_frequencies_cut).T

        # Feature transformation(standardization or others).
        return self.transformation.transform(result) if self.transformation is not None else result

    @property
    def inputs(self):
        return DataSubSet(data_key=self.inputs_key, complete_data_set=self)

    @property
    def labels(self):
        return self.valid_samples[:, self.labels_key]

    def sample_at(self, index):
        onset, label = self.valid_samples[index]
        inputs = self._transform_raw_signal_data(self.eeg_signals[self.picks, onset:onset + self.steps_per_sample][0])
        return inputs, label

    def remove_samples(self, samples_indices):
        self.valid_samples = np.delete(self.valid_samples, samples_indices, 0)

    def _replace_labels(self, mapping, samples_indices=None):
        if samples_indices is None:
            self.valid_samples = np.array([(onset, mapping.get(label, label))
                                           for onset, label in self.valid_samples])
        else:
            self.valid_samples = np.array([(onset, mapping.get(label, label) if index in samples_indices else label)
                                           for index, (onset, label) in enumerate(self.valid_samples)])


class FmedLfaEegExperimentData(EpochEegExperimentDataSet):

    @property
    def signal_classification(self):
        import scipy.io
        mat_filename = list_folder_files_with_extension(self.files_folder_path, 'mat')[0]
        labels_data = scipy.io.loadmat(mat_filename)['stageData'][0][0]

        # Select the index of the stages in de matrix, most of the time is 5 but sometimes it's 6.
        stage_info_index = 5 if labels_data[5].dtype == np.dtype('uint8') else 6
        onset_index = stage_info_index + 1

        return list(zip(labels_data[onset_index][:, 0], labels_data[stage_info_index][:, 0]))


class FmedLfaExperimentDataSet(SequentialDataSetsMerge):

    def __init__(self, experiments_data_folders, epoch_duration, **kwargs):
        eeg_experiments_data = [FmedLfaEegExperimentData(experiments_data_folder, epoch_duration, **kwargs)
                                for experiments_data_folder
                                in experiments_data_folders]
        super().__init__(eeg_experiments_data,
                         inputs_key=eeg_experiments_data[0].inputs_key,
                         labels_key=eeg_experiments_data[0].labels_key,
                         features_names=eeg_experiments_data[0].features_names,
                         **kwargs)