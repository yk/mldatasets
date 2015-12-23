import random
import zipfile
import bz2
import gzip
import requests
import os
import numpy as np
import sklearn.datasets
from sklearn import preprocessing


def download_file(url: str) -> str:
    resp = requests.get(url)
    return resp.text


def download_binary_file(url: str):
    resp = requests.get(url)
    return resp.content


def save_file(content: str, name: str):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    with open(name, "w") as file:
        file.write(content)


def save_binary_file(content: bytes, name: str):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    with open(name, "wb") as file:
        file.write(content)


def read_sparse_vector(tokens, dimensions):
    vec = np.zeros(dimensions)
    for token in tokens:
        parts = token.split(":")
        position = int(parts[0]) - 1
        value = float(parts[1])
        vec[position] = value
    return vec


class Dataset:
    base_dir = os.path.expanduser("~/tmp/pydata")
    cache_dir = "{}/cache".format(base_dir)

    def get_name(self):
        return self.__class__.__name__

    def get_task(self):
        pass

    def is_available(self) -> bool:
        pass

    def make_available(self):
        pass

    def convert(self) -> (np.ndarray, np.ndarray):
        pass

    def cache_file_name(self, postfix: str = "") -> str:
        return "{}/{}.{}.npy".format(Dataset.cache_dir, self.get_name(), postfix)

    def cache_write(self, data: np.ndarray, labels: np.ndarray):
        np.save(self.cache_file_name("data"), data)
        np.save(self.cache_file_name("labels"), labels)

    def cache_read(self) -> (np.ndarray, np.ndarray):
        return np.load(self.cache_file_name("data")), np.load(self.cache_file_name("labels"))

    def cache_available(self) -> bool:
        return os.path.exists(self.cache_file_name("data")) and os.path.exists(self.cache_file_name("labels"))

    def get_data(self, force_write_cache=False) -> (np.ndarray, np.ndarray):
        if force_write_cache or not self.cache_available():
            if not self.is_available():
                self.make_available()
            data, labels = self.convert()
            self.cache_write(data, labels)
            return data, labels
        else:
            return self.cache_read()

    def __str__(self):
        return self.get_name()


class NoCacheDataset(Dataset):
    def cache_available(self):
        return False

    def cache_write(self, data: np.ndarray, labels: np.ndarray):
        pass


class DataProvider:
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data, self.labels = data, labels
        self.size = self.data.shape[0]
        self.dimensions = self.data.shape[1]

    def get_sample(self, sample_size: int = 50) -> (np.ndarray, np.ndarray):
        d, l, i = self.get_sample_with_inds(sample_size)
        return d, l

    def get_sample_with_inds(self, sample_size: int = 50) -> (np.ndarray, np.ndarray, list):
        data_inds = range(self.size)
        sample_inds = random.sample(data_inds, sample_size)
        return self.data[sample_inds, :], self.labels[sample_inds], sample_inds

    def zero_point(self):
        return np.zeros_like(self.data[0])


def create_data_provider(dataset: Dataset, force_write_cache: bool = False, center_data: bool = True,
                         scale_data: bool = True, add_bias_feature: bool = True, normalize_datapoints: bool = True,
                         center_labels: bool = False, scale_labels: bool = False,
                         transform_labels_to_plus_minus_one: bool = True):
    data, labels = dataset.get_data(force_write_cache=force_write_cache)
    copy=False
    if scale_data:
        preprocessing.scale(data, copy=copy)
    elif center_data:
        data = preprocessing.scale(data, with_std=False, copy=copy)
    if scale_labels:
        labels = preprocessing.scale(labels, copy=copy)
    elif center_labels:
        labels = preprocessing.scale(labels, with_std=False, copy=copy)
    if add_bias_feature:
        data = np.hstack((data, np.ones((data.shape[0], 1))))
    if normalize_datapoints:
        data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    if transform_labels_to_plus_minus_one:
        labels = labels * 2.0 - 1.0
    return DataProvider(data, labels)


class ClassificationDataset(Dataset):
    def get_task(self):
        return "classification"


class MultiLabelClassificationDataset(Dataset):
    def get_task(self):
        return "multilabel_classification"


class RegressionDataset(Dataset):
    def get_task(self):
        return "regression"


class RosenbrockBanana(Dataset):
    def __init__(self, dimensions: int = 200, size: int = 1000):
        super().__init__()
        self.size = size
        self.dimensions = dimensions

    def is_available(self):
        return True

    def get_task(self):
        return "rosenbrock"

    def convert(self):
        return np.array([[1.0] * self.dimensions] * self.size), np.array([0.0] * self.size)


class QuadraticDataset(Dataset):
    def __init__(self, diag: np.ndarray, size: int = 1000):
        super().__init__()
        self.size = size
        self.diag = diag
        self.dimensions = len(diag)

    def is_available(self):
        return True

    def get_task(self):
        return "quadratic"

    def convert(self):
        return np.array([self.diag] * self.size), np.array([0.0] * self.size)


class SingleFileOnlineDataset(Dataset):
    def __init__(self, url: str, filename: str, dimensions: int):
        self.url = url
        self.filename = filename
        self.dimensions = dimensions

    def is_available(self) -> bool:
        return os.path.isfile(self.filename)

    def make_available(self):
        save_file(download_file(self.url), self.filename)

    def convert_line(self, line: str) -> (np.ndarray, float):
        pass

    def convert(self):
        vectors = []
        classes = []
        with open(self.filename) as file:
            for line in file:
                if line.isspace():
                    continue
                vec, cls = self.convert_line(line.strip())
                vectors.append(vec)
                classes.append(cls)
        return np.array(vectors), np.array(classes)


class MultipleFilesOnlineDataset(Dataset):
    def __init__(self, urls: list, filenames: list, dimensions: int):
        self.urls = urls
        self.filenames = filenames
        self.dimensions = dimensions

    def is_available(self) -> bool:
        for fn in self.filenames:
            if not os.path.isfile(fn):
                return False
        return True

    def make_available(self):
        for url, fn in zip(self.urls, self.filenames):
            save_file(download_file(url), fn)

    def convert_lines(self, lines: list) -> (np.ndarray, float):
        pass

    def convert(self):
        files = []
        for fn in self.filenames:
            with open(fn) as file:
                lines = [l.strip() for l in file.readlines()]
                files.append(lines)
        vectors = []
        classes = []
        for lines in zip(*files):
            vec, cls = self.convert_lines(lines)
            vectors.append(vec)
            classes.append(cls)
        return np.array(vectors), np.array(classes)


class UCIMLAdult(SingleFileOnlineDataset, ClassificationDataset):
    def __init__(self, name: str, dimensions: int):
        self.name = name
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(self.name),
                         filename="{}/uciml/{}".format(Dataset.base_dir, self.name), dimensions=dimensions)

    def get_label(self, labelstr: str) -> float:
        pass

    def convert_line(self, line: str) -> (np.ndarray, float):
        tokens = line.split()
        cls = self.get_label(tokens[0])
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Mushrooms(UCIMLAdult):
    def __init__(self):
        super().__init__("mushrooms", 112)

    def get_label(self, labelstr: str) -> float:
        return float(labelstr) - 1.0


class A9A(UCIMLAdult):
    def __init__(self):
        super().__init__("a9a", 123)

    def get_label(self, labelstr: str) -> float:
        return float((int(labelstr) + 1) / 2)


class Gisette(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=6000, dimensions=5000):
        super().__init__(
            urls=["http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data",
                  "http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"],
            filenames=["{}/gisette.data".format(Dataset.base_dir), "{}/gisette.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines: list):
        vec = np.array([float(t) for t in lines[0].split()[:self.dimensions]])
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class Dexter(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=2600, dimensions=20000):
        super().__init__(
            urls=["http://archive.ics.uci.edu/ml/machine-learning-databases/dexter/DEXTER/dexter_train.data",
                  "http://archive.ics.uci.edu/ml/machine-learning-databases/dexter/DEXTER/dexter_train.labels"],
            filenames=["{}/dexter.data".format(Dataset.base_dir), "{}/dexter.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines: list):
        vec = read_sparse_vector(lines[0].split(), 20000)[:self.dimensions]
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class Arcene(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=900, dimensions=10000):
        super().__init__(
            urls=["http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data",
                  "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels"],
            filenames=["{}/arcene.data".format(Dataset.base_dir), "{}/arcene.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines: list):
        vec = np.array([float(t) for t in lines[0].split()[:self.dimensions]])
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class BZ2Dataset(SingleFileOnlineDataset):
    def make_available(self):
        bz2filename = "{}.bz2".format(self.filename)
        if not os.path.isfile(bz2filename):
            save_binary_file(download_binary_file(self.url), bz2filename)
        with bz2.open(bz2filename, 'r') as f:
            data = f.read().decode("ascii")
            save_file(data, self.filename)


class Ijcnn1(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
                         filename="{}/ijcnn1".format(Dataset.base_dir), dimensions=22)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = float((int(tokens[0]) + 1) / 2)
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Covtype(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
                         filename="{}/covtype".format(Dataset.base_dir), dimensions=54)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = float((int(tokens[0]) - 1))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class MNIST(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2",
                         filename="{}/mnist".format(Dataset.base_dir), dimensions=780)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = 0.0 if int(tokens[0]) <= 4 else 1.0
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class YearPredictionMSD(BZ2Dataset, RegressionDataset):
    def __init__(self):
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
                         filename="{}/yearprediction".format(Dataset.base_dir), dimensions=90)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = float(int(tokens[0]))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class ZipDataset(SingleFileOnlineDataset):
    def __init__(self, url: str, filename: str, dimensions: int, filename_in_zip: str):
        super().__init__(url, filename, dimensions)
        self.filename_in_zip = filename_in_zip

    def make_available(self):
        zipfilename = "{}.zip".format(self.filename)
        if not os.path.isfile(zipfilename):
            save_binary_file(download_binary_file(self.url), zipfilename)
        with zipfile.ZipFile(zipfilename) as zfile:
            with zfile.open(self.filename_in_zip, 'r') as f:
                data = f.read().decode("ascii")
                save_file(data, self.filename)


class BlogFeedback(ZipDataset, RegressionDataset):
    def __init__(self):
        super().__init__(url="https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip",
                         filename="{}/blogfeedback".format(Dataset.base_dir), dimensions=280,
                         filename_in_zip="blogData_train.csv")

    def convert_line(self, line: str):
        tokens = line.split(r",")
        cls = float(tokens[-1])
        vec = np.array([float(t) for t in tokens[:-1]])
        return vec, cls


class StanfordSpam(SingleFileOnlineDataset, ClassificationDataset):
    def __init__(self):
        super().__init__(url="http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data",
                         filename="{}/spam".format(Dataset.base_dir), dimensions=67)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = 1.0 if tokens[-1] == "1" else 0.0
        vec = np.array([float(t) for t in tokens[:-1]])
        return vec, cls


class CpuSmall(SingleFileOnlineDataset, RegressionDataset):
    def __init__(self):
        super().__init__(url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale",
                         filename="{}/cpusmall".format(Dataset.base_dir), dimensions=12)

    def convert_line(self, line: str):
        tokens = line.split()
        cls = float(int(tokens[0]))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Diabetes(RegressionDataset):
    def is_available(self):
        return True

    def convert(self):
        bunch = sklearn.datasets.load_diabetes()
        return bunch["data"], bunch["target"]


class GaussianNoiseRegressionGenerated(NoCacheDataset, RegressionDataset):
    def __init__(self, w: np.ndarray, sigma: float, size: int):
        self.w = np.asarray(w)
        self.sigma = sigma
        self.size = size

    def is_available(self):
        return True

    def convert(self):
        d = self.w.shape[0]
        x = np.random.random((self.size, d))
        xw = np.dot(x, self.w)
        y = xw + np.random.randn(self.size) * self.sigma
        return x, y


class MNISTFull(MultiLabelClassificationDataset):
    def __init__(self):
        self.source_url = 'http://yann.lecun.com/exdb/mnist'
        self.data_url = '{}/train-images-idx3-ubyte.gz'.format(self.source_url)
        self.labels_url = '{}/train-labels-idx1-ubyte.gz'.format(self.source_url)
        self.data_filename = "{}/mnist_full.data".format(Dataset.base_dir)
        self.labels_filename = "{}/mnist_full.labels".format(Dataset.base_dir)

    def is_available(self):
        return os.path.exists(self.data_filename) and os.path.exists(self.labels_filename)

    def make_available(self):
        save_binary_file(download_binary_file(self.data_url), self.data_filename)
        save_binary_file(download_binary_file(self.labels_url), self.labels_filename)

    def convert(self):
        images = self.extract_images(self.data_filename)
        labels = self.extract_labels(self.labels_filename)
        data = np.reshape(images, [images.shape[0], -1])
        data = data.astype(np.float32)/255.0
        labels = labels.astype(np.int32)
        return data, labels

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def extract_labels(self, filename, one_hot=False):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self.dense_to_one_hot(labels)
            return labels



# make dirs
if not os.path.exists(Dataset.cache_dir):
    os.makedirs(Dataset.cache_dir, mode=0o775, exist_ok=True)

if __name__ == '__main__':
    d1 = GaussianNoiseRegressionGenerated([1., 2.], 1., 5)
    print(d1.get_data())

    d2 = MNISTFull().get_data()
    print(d2[0].shape)
    print(d2[0][0].dtype)
    print(d2[0][0])
    print(d2[1][0:10])
