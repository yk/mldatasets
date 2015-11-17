from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import random
import zipfile
import bz2
import requests
import os
import numpy as np
import sklearn
from io import open
from itertools import izip


def download_file(url):
    resp = requests.get(url)
    return resp.text


def download_binary_file(url):
    resp = requests.get(url)
    return resp.content


def save_file(content, name):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    with open(name, u"w") as file:
        file.write(content)


def save_binary_file(content, name):
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    with open(name, u"wb") as file:
        file.write(content)


def read_sparse_vector(tokens, dimensions):
    vec = np.zeros(dimensions)
    for token in tokens:
        parts = token.split(u":")
        position = int(parts[0]) - 1
        value = float(parts[1])
        vec[position] = value
    return vec


class Dataset(object):
    base_dir = os.path.expanduser(u"~/tmp/pydata")

    def get_name(self):
        return self.__class__.__name__

    def get_task(self):
        pass

    def is_available(self):
        pass

    def make_available(self):
        pass

    def convert(self):
        pass

    def get_data(self):
        if not self.is_available():
            self.make_available()
        return self.convert()

    def __str__(self):
        return self.get_name()


class DataProvider(object):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
        self.size = self.data.shape[0]
        self.dimensions = self.data.shape[1]

    def get_sample(self, sample_size=50):
        d, l, i = self.get_sample_with_inds(sample_size)
        return d, l

    def get_sample_with_inds(self, sample_size=50):
        data_inds = xrange(self.size)
        sample_inds = random.sample(data_inds, sample_size)
        return self.data[sample_inds, :], self.labels[sample_inds], sample_inds

    def zero_point(self):
        return np.zeros_like(self.data[0])


class ClassificationDataset(Dataset):
    def get_task(self):
        return u"classification"


class RegressionDataset(Dataset):
    def get_task(self):
        return u"regression"


class RosenbrockBanana(Dataset):
    def __init__(self, dimensions=200, size=1000):
        super(RosenbrockBanana, self).__init__()
        self.size = size
        self.dimensions = dimensions

    def is_available(self):
        return True

    def get_task(self):
        return u"rosenbrock"

    def convert(self):
        return np.array([[1.0] * self.dimensions] * self.size), np.array([0.0] * self.size)


class QuadraticDataset(Dataset):
    def __init__(self, diag, size=1000):
        super(QuadraticDataset, self).__init__()
        self.size = size
        self.diag = diag
        self.dimensions = len(diag)

    def is_available(self):
        return True

    def get_task(self):
        return u"quadratic"

    def convert(self):
        return np.array([self.diag]*self.size), np.array([0.0]*self.size)


class SingleFileOnlineDataset(Dataset):
    def __init__(self, url, filename, dimensions):
        self.url = url
        self.filename = filename
        self.dimensions = dimensions

    def is_available(self):
        return os.path.isfile(self.filename)

    def make_available(self):
        save_file(download_file(self.url), self.filename)

    def convert_line(self, line):
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
    def __init__(self, urls, filenames, dimensions):
        self.urls = urls
        self.filenames = filenames
        self.dimensions = dimensions

    def is_available(self):
        for fn in self.filenames:
            if not os.path.isfile(fn):
                return False
        return True

    def make_available(self):
        for url, fn in izip(self.urls, self.filenames):
            save_file(download_file(url), fn)

    def convert_lines(self, lines):
        pass

    def convert(self):
        files = []
        for fn in self.filenames:
            with open(fn) as file:
                lines = [l.strip() for l in file.readlines()]
                files.append(lines)
        vectors = []
        classes = []
        for lines in izip(*files):
            vec, cls = self.convert_lines(lines)
            vectors.append(vec)
            classes.append(cls)
        return np.array(vectors), np.array(classes)


class UCIMLAdult(SingleFileOnlineDataset, ClassificationDataset):
    def __init__(self, name, dimensions):
        self.name = name
        super(UCIMLAdult, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(self.name),
                         filename=u"{}/uciml/{}".format(Dataset.base_dir, self.name), dimensions=dimensions)

    def get_label(self, labelstr):
        pass

    def convert_line(self, line):
        tokens = line.split()
        cls = self.get_label(tokens[0])
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Mushrooms(UCIMLAdult):
    def __init__(self):
        super(Mushrooms, self).__init__(u"mushrooms", 112)

    def get_label(self, labelstr):
        return float(labelstr) - 1.0


class A9A(UCIMLAdult):
    def __init__(self):
        super(A9A, self).__init__(u"a9a", 123)

    def get_label(self, labelstr):
        return float((int(labelstr) + 1) / 2)


class Gisette(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=6000, dimensions=5000):
        super(Gisette, self).__init__(
            urls=[u"http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data",
                  u"http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"],
            filenames=[u"{}/gisette.data".format(Dataset.base_dir), u"{}/gisette.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines):
        vec = np.array([float(t) for t in lines[0].split()[:self.dimensions]])
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class Dexter(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=2600, dimensions=20000):
        super(Dexter, self).__init__(
            urls=[u"http://archive.ics.uci.edu/ml/machine-learning-databases/dexter/DEXTER/dexter_train.data",
                  u"http://archive.ics.uci.edu/ml/machine-learning-databases/dexter/DEXTER/dexter_train.labels"],
            filenames=[u"{}/dexter.data".format(Dataset.base_dir), u"{}/dexter.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines):
        vec = read_sparse_vector(lines[0].split(), 20000)[:self.dimensions]
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class Arcene(MultipleFilesOnlineDataset, ClassificationDataset):
    def __init__(self, size=900, dimensions=10000):
        super(Arcene, self).__init__(
            urls=[u"http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data",
                  u"http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels"],
            filenames=[u"{}/arcene.data".format(Dataset.base_dir), u"{}/arcene.labels".format(Dataset.base_dir)],
            dimensions=dimensions)
        self.size = size

    def convert_lines(self, lines):
        vec = np.array([float(t) for t in lines[0].split()[:self.dimensions]])
        cls = float((int(lines[1]) + 1) / 2)
        return vec, cls


class BZ2Dataset(SingleFileOnlineDataset):
    def make_available(self):
        bz2filename = u"{}.bz2".format(self.filename)
        if not os.path.isfile(bz2filename):
            save_binary_file(download_binary_file(self.url), bz2filename)
        with bz2.open(bz2filename, u'r') as f:
            data = f.read().decode(u"ascii")
            save_file(data, self.filename)


class Ijcnn1(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super(Ijcnn1, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
                         filename=u"{}/ijcnn1".format(Dataset.base_dir), dimensions=22)

    def convert_line(self, line):
        tokens = line.split()
        cls = float((int(tokens[0]) + 1) / 2)
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Covtype(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super(Covtype, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
                         filename=u"{}/covtype".format(Dataset.base_dir), dimensions=54)

    def convert_line(self, line):
        tokens = line.split()
        cls = float((int(tokens[0]) - 1))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class MNIST(BZ2Dataset, ClassificationDataset):
    def __init__(self):
        super(MNIST, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2",
                         filename=u"{}/mnist".format(Dataset.base_dir), dimensions=780)

    def convert_line(self, line):
        tokens = line.split()
        cls = 0.0 if int(tokens[0]) <= 4 else 1.0
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class YearPredictionMSD(BZ2Dataset, RegressionDataset):
    def __init__(self):
        super(YearPredictionMSD, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
                         filename=u"{}/yearprediction".format(Dataset.base_dir), dimensions=90)

    def convert_line(self, line):
        tokens = line.split()
        cls = float(int(tokens[0]))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class ZipDataset(SingleFileOnlineDataset):
    def __init__(self, url, filename, dimensions, filename_in_zip):
        super(ZipDataset, self).__init__(url, filename, dimensions)
        self.filename_in_zip = filename_in_zip

    def make_available(self):
        zipfilename = u"{}.zip".format(self.filename)
        if not os.path.isfile(zipfilename):
            save_binary_file(download_binary_file(self.url), zipfilename)
        with zipfile.ZipFile(zipfilename) as zfile:
            with zfile.open(self.filename_in_zip, u'r') as f:
                data = f.read().decode(u"ascii")
                save_file(data, self.filename)


class BlogFeedback(ZipDataset, RegressionDataset):
    def __init__(self):
        super(BlogFeedback, self).__init__(url=u"https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip",
                         filename=u"{}/blogfeedback".format(Dataset.base_dir), dimensions=280,
                         filename_in_zip=u"blogData_train.csv")

    def convert_line(self, line):
        tokens = line.split(ur",")
        cls = float(tokens[-1])
        vec = np.array([float(t) for t in tokens[:-1]])
        return vec, cls


class StanfordSpam(SingleFileOnlineDataset, ClassificationDataset):
    def __init__(self):
        super(StanfordSpam, self).__init__(url=u"http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data",
                         filename=u"{}/spam".format(Dataset.base_dir), dimensions=67)

    def convert_line(self, line):
        tokens = line.split()
        cls = 1.0 if tokens[-1] == u"1" else 0.0
        vec = np.array([float(t) for t in tokens[:-1]])
        return vec, cls


class CpuSmall(SingleFileOnlineDataset, RegressionDataset):
    def __init__(self):
        super(CpuSmall, self).__init__(url=u"http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale",
                         filename=u"{}/cpusmall".format(Dataset.base_dir), dimensions=12)

    def convert_line(self, line):
        tokens = line.split()
        cls = float(int(tokens[0]))
        vec = read_sparse_vector(tokens[1:], self.dimensions)
        return vec, cls


class Diabetes(RegressionDataset):
    def is_available(self):
        return True

    def convert(self):
        bunch = sklearn.datasets.load_diabetes()
        return bunch[u"data"], bunch[u"target"]
