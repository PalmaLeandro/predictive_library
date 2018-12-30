import logging
import random
import numpy as np

from sklearn.model_selection import train_test_split


class DataSet(object):

    @property
    def inputs(self):
        pass

    @property
    def labels(self):
        pass

    @property
    def labels_names(self):
        pass

    def get_all_samples(self):
        pass

    def get_sample(self):
        pass

    def get_sample_batch(self, batch_size):
        pass

    def provide_train_validation_random_partition(self):
        pass

    def balance_classes(self, **kwargs):
        pass

    def add_samples(self, inputs, labels, **kwargs):
        pass

    def blend_classes(self, classes_to_blend, **kwargs):
        pass


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

        ax.bar(classes_histogram_df['class'], classes_histogram_df['num_samples'])
        plt.xticks(rotation=90)

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


class ArrayDataSet(DataSet):

    def __init__(self, inputs, labels=np.array([]), test_proportion=None, validation_proportion=None, labels_names=None,
                 exact_set_division=True, inputs_key=0, labels_key=1, **kwargs):
        self.validation_proportion = validation_proportion
        self._labels_names = labels_names
        self.exact_set_division = exact_set_division
        self.inputs_key = inputs_key
        self.labels_key = labels_key
        if test_proportion is None:
            self._train_inputs, self._train_labels = inputs, labels
            self._test_inputs, self._test_labels = inputs, labels
        else:
            self._train_inputs, \
            self._test_inputs, \
            self._train_labels, \
            self._test_labels = divide_inputs_labels_set_randomly(inputs, labels, test_proportion, exact_set_division,
                                                                  inputs_key, labels_key)

    @property
    def labels_names(self):
        return self._labels_names if self._labels_names is not None \
               else [str(label) for label in range(self.num_classes)]

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
        return len(self._test_inputs)

    @property
    def test_proportion(self):
        return self.num_test_samples / float(self.num_test_samples + self.num_train_samples)

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
        if self._train_inputs is self._test_inputs:
            return self._train_inputs
        else:
            return np.concatenate([self.train_inputs, self.test_inputs])

    @property
    def labels(self):
        if self._train_labels is self._test_labels:
            return self._train_labels
        else:
            return np.concatenate([self.train_labels, self.test_labels])

    def get_all_samples(self):
        if self._train_inputs is self._test_inputs and self._train_labels is self._test_labels:
            return self.train_inputs, self._train_labels
        else:
            return (np.concatenate([self._train_inputs, self._test_inputs]),
                    np.concatenate([self._train_labels, self._test_labels]))

    def get_sample(self):
        return sample_input_label(*self.get_all_samples())

    def get_train_sample(self):
        return sample_input_label(self._train_inputs, self._train_labels)

    def get_test_sample(self):
        return sample_input_label(self._test_inputs, self._test_labels)

    def classes_distribution(self, show_chart=True):
        classes_histogram_df, ax = dataset_class_distribution(self.labels, self._labels_names)
        ax.set_title('Dataset classes distribution')
        if show_chart:
            import matplotlib.pyplot as plt
            plt.show()

        return classes_histogram_df

    def train_test_classes_distribution(self):
        import pandas as pd
        import matplotlib.pyplot as plt

        _, (train_dataset_classes_ax, test_dataset_classes_ax) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

        train_samples_histogram_df, _ = dataset_class_distribution(self.train_labels, self._labels_names,
                                                                   show_table=False,
                                                                   ax=train_dataset_classes_ax)
        plt.sca(train_dataset_classes_ax)
        plt.xticks(rotation=-90)
        train_dataset_classes_ax.set_title('Train dataset classes distribution')

        test_samples_histogram_df, _ = dataset_class_distribution(self.test_labels, self._labels_names,
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
        test_proportion = self.test_proportion

        inputs_to_add = []
        labels_to_add = []
        samples_indices_to_remove = []
        adds_and_removals_to_do = np.sum(np.apply_along_axis(abs, 0, classes_num_samples_to_vary))
        while any([num_missing_samples != 0 for num_missing_samples in classes_num_samples_to_vary]):
            sample_from_test = random_choice(test_proportion)
            class_input, \
            class_label, \
            sample_index = sample_input_label(self.test_inputs if sample_from_test else self.train_inputs,
                                              self.test_labels if sample_from_test else self.train_labels,
                                              return_sample_index=True)

            if classes_num_samples_to_vary[class_label] > 0:
                inputs_to_add.append(class_input)
                labels_to_add.append(class_label)
                classes_num_samples_to_vary[class_label] -= 1
            else:
                if classes_num_samples_to_vary[class_label] < 0:
                    samples_indices_to_remove.append(sample_index)
                    classes_num_samples_to_vary[class_label] += 1
            progress = float(np.sum(np.apply_along_axis(abs, 0, classes_num_samples_to_vary))) / adds_and_removals_to_do
            progress = (1 - progress) * 100
            if int(progress) % 10 == 0:
                logging.info('Modifiying dataset distribution: {progress}% progress.'.format(progress=progress))

        if len(inputs_to_add) > 0 and len(labels_to_add) > 0:
            self.add_samples(np.array(inputs_to_add), np.array(labels_to_add))
        if len(samples_indices_to_remove) > 0:
            self.remove_samples(samples_indices_to_remove)

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

    def replace_labels(self, mapping):
        self._train_labels = np.array([mapping.get(label, label) for label in self._train_labels])
        self._test_labels = np.array([mapping.get(label, label) for label in self._test_labels])

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

        self.replace_labels(mapping)

        blend_name = blend_name if blend_name is not None \
            else ' + '.join([self.labels_names[class_label] for class_label in classes_to_blend])

        self._labels_names[blend_label] = blend_name
        self._labels_names = [label_name for class_label, label_name in enumerate(self.labels_names)
                            if class_label not in classes_to_blend or class_label == blend_label]

    def __getitem__(self, key):
        if np.isscalar(key):
            return self.sample_at(key)
        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                if key[1] == self.labels_key:
                    return self.labels[key[0]]
                if key[1] == self.inputs_key:
                    return self.inputs


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


def data_partition(data_sets, key):
    return DataSetsMerge([data_set[:, key] for data_set in data_sets])


def validate_train_labels(labels):
    labels = labels.tolist()
    if np.max(labels) != len(np.unique(labels)) - 1:
        for label in range(max(np.max(labels), len(np.unique(labels)))):
            if label not in np.unique(labels):
                labels = np.append(labels, np.array([label]))
    return labels


def coarse_subsets_partition(subsets, ratio, shuffle=True):
    if ratio not in (0, 1, None):
        if shuffle:
            import random
            random.shuffle(subsets)
        subsets_lengths = [len(subset) for subset in subsets]
        subsets_cumulative_lengths = np.cumsum(subsets_lengths)
        total_length = subsets_cumulative_lengths[-1]
        cut_index = min(*([index for index, cumulative_length in enumerate(subsets_cumulative_lengths)
                           if cumulative_length > (total_length * ratio)] + [len(subsets_cumulative_lengths) - 1]))
        A = subsets[:cut_index]
        B = subsets[cut_index:]
    else:
        A = subsets
        B = []

    return A, B


class DataSetsMerge(ArrayDataSet):

    def __init__(self, data_sets, test_proportion=0., validation_proportion=None, labels_names=None,
                 exact_merge=False, inputs_key=0, labels_key=1, **kwargs):
        self.data_sets = data_sets
        self.exact_merge = exact_merge
        self.inputs_key = inputs_key
        self.labels_key = labels_key
        self._labels_names = labels_names

        if exact_merge:
            super(SequentialData, self).__init__(inputs=KeyPartialSequentialData(inputs_key,
                                                                                 complete_sequential_data=self),
                                                 labels=KeyPartialSequentialData(labels_key,
                                                                                 complete_sequential_data=self),
                                                 test_proportion=test_proportion,
                                                 validation_proportion=validation_proportion,
                                                 labels_names=labels_names, **kwargs)
        else:
            train_sequences_sets, test_sequences_sets = coarse_subsets_partition(data_sets,
                                                                                 1 - test_proportion)
            self.num_train_data_sets = len(train_sequences_sets)
            self.data_sets = train_sequences_sets + test_sequences_sets

    @property
    def train_inputs(self):
        return data_partition(self.data_sets[:self.num_train_data_sets], self.inputs_key)

    @property
    def train_labels(self):
        return data_partition(self.data_sets[:self.num_train_data_sets], self.labels_key)

    @property
    def test_inputs(self):
        return data_partition(self.data_sets[self.num_train_data_sets:], self.inputs_key)

    @property
    def test_labels(self):
        return data_partition(self.data_sets[self.num_train_data_sets:], self.labels_key)

    @property
    def inputs(self):
        return data_partition(self.data_sets, self.inputs_key)

    @property
    def labels(self):
        return data_partition(self.data_sets, self.labels_key)

    def provide_train_validation_random_partition(self, validation_proportion=None):
        '''For training and validating by bootstraping(sampling with repetitions).'''
        validation_proportion = validation_proportion or self.validation_proportion
        if self.exact_merge:
            return super(DataSetsMerge, self).provide_train_validation_random_partition(validation_proportion)
        else:
            train_sequence_sets = self.data_sets[:self.num_train_data_sets]

            train_fold, validation_fold = divide_set_randomly(train_sequence_sets, validation_proportion)

            train_fold_inputs = DataSetsMerge([data_set[:, self.inputs_key]
                                               for data_set in train_fold],
                                              test_proportion=0, validation_proportion=0,
                                              labels_names=self.labels_names, exact_merge=False,
                                              inputs_key=self.inputs_key, labels_key=self.labels_key)
            train_fold_labels = DataSetsMerge([data_set[:, self.labels_key]
                                               for data_set in train_fold],
                                              test_proportion=1, validation_proportion=0,
                                              labels_names=self.labels_names, exact_merge=False,
                                              inputs_key=self.inputs_key, labels_key=self.labels_key)

            validation_fold_inputs = DataSetsMerge([data_set[:, self.inputs_key]
                                                    for data_set in validation_fold],
                                                   test_proportion=0, validation_proportion=1,
                                                   labels_names=self.labels_names, exact_merge=False,
                                                   inputs_key=self.inputs_key, labels_key=self.labels_key)
            validation_fold_labels = DataSetsMerge([data_set[:, self.labels_key]
                                                    for data_set in validation_fold],
                                                   test_proportion=1, validation_proportion=1,
                                                   labels_names=self.labels_names, exact_merge=False,
                                                   inputs_key=self.inputs_key, labels_key=self.labels_key)

            return train_fold_inputs, validation_fold_inputs, train_fold_labels, validation_fold_labels

    @property
    def num_samples(self):
        return sum([len(data_set) for data_set in self.data_sets])

    @property
    def num_train_samples(self):
        return sum([data_set.num_samples
                    for data_set
                    in self.data_sets[:self.num_train_data_sets]])

    @property
    def num_test_samples(self):
        return sum([data_set.num_samples
                    for data_set
                    in self.data_sets[self.num_train_data_sets:]])

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

    def __getitem__(self, key):
        if np.isscalar(key):
            return self.sample_at(key)
        else:
            return [self.sample_at(index) for index in np.array(range(self.num_samples))[key]]

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
        for data_set, data_set_offset, samples_indices_to_remove in zip(self.data_sets,
                                                                        [0] + cumulative_data_sets_length[:-1].tolist(),
                                                                        data_sets_samples):
            data_set.remove_samples(np.array(samples_indices_to_remove) - data_set_offset)

    def replace_labels(self, mapping):
        for data_set in self.data_sets:
            data_set.replace_labels(mapping)


class SequentialDataSetsMerge(DataSetsMerge, SequentialData):

    @property
    def steps_per_sample(self):
        return self.data_sets[0].shape[-2]


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
        self.valid_samples = np.array([(onset, label)
                                       for onset_index, (onset, label) in enumerate(signal_classification)
                                       if onset_index + 1 < len(signal_classification)
                                          and (signal_classification[onset_index + 1][0] - onset) == steps_per_sample])
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

    @property
    def inputs(self):
        return KeyPartialSequentialData(data_key=self.inputs_key, complete_sequential_data=self)

    @property
    def labels(self):
        return self.valid_samples[:, self.labels_key]

    def sample_at(self, index):
        onset, label = self.valid_samples[index]
        # Omit empty stim channel and time dimmension.
        return self._transform_raw_signal_data(self.eeg_signals[:-1, onset:onset + self.steps_per_sample][0]), label

    def remove_samples(self, samples_indices):
        self.valid_samples = np.delete(self.valid_samples, samples_indices, 0)

    def replace_labels(self, mapping):
        self.valid_samples = np.array([(onset, mapping.get(label, label)) for onset, label in self.valid_samples])


class FmedLfaEegExperimentData(EpochEegExperimentData):

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
        super().__init__(eeg_experiments_data, inputs_key=eeg_experiments_data[0].inputs_key,
                         labels_key=eeg_experiments_data[0].labels_key, **kwargs)