from __future__ import absolute_import, division, print_function

import gzip
import io

import apache_beam as beam
import IPython.display as display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python import datasets
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated

tf.enable_eager_execution()


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return tf.one_hot(labels, num_classes)
        return labels


def get_images_and_labels(images_path, labels_path):
    """
    Extract gzip images/labels from path
    """
    with gfile.Open(images_path, 'rb') as f:
        images = extract_images(f)

    with gfile.Open(labels_path, 'rb') as f:
        labels = extract_labels(f)

    return images, labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




images_path = '/tmp/data/mnist/val/images.gz'
labels_path = '/tmp/data/mnist/val/labels.gz'

val_images, val_labels = get_images_and_labels(images_path, labels_path)

def get_images_and_labels_w_index(images, labels):
    """
    Attach indexes to each record to be used for `beam.CoGroupByKey`

    TODO: also to be used by `beam._partition_fn` in the future
    """
    images_w_index = [(i, x) for i, x in enumerate(images)]
    labels_w_index = [(i, x) for i, x in enumerate(labels)]
    return images_w_index, labels_w_index


images_w_index, labels_w_index = get_images_and_labels_w_index(val_images, val_labels)


def get_nhwc(images):
    """
    Returns NHWC to use for creating `tf.train.Example`
    """
    N = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    return N, rows, cols, depth


N, rows, cols, depth = get_nhwc(val_images)


def group_by_tf_example(key_value):
    _, value = key_value
    image = value['image'][0]
    label = value['label'][0]
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image.tostring())
        }))
    return example


# images and labels combined and written to TFRecords gzip file

tfrecord_outfile = 'mnist-out'
file_name_suffix = '.gz'

with beam.Pipeline(options=PipelineOptions()) as p:
    label_line = p | "CreateLabel" >> beam.Create(labels_w_index[:1])
    image_line = p | "CreateImage" >> beam.Create(images_w_index[:1])

    group_by = ({
        'label': label_line,
        'image': image_line
    }) | beam.CoGroupByKey()

    tf_example = group_by | "GroupByToTfExample" >> beam.Map(
        group_by_tf_example)

    serialize = (tf_example | 'SerializeDeterministically' >>
                 beam.Map(lambda x: x.SerializeToString(deterministic=True)))

    output = serialize | beam.io.WriteToTFRecord(tfrecord_outfile,
                                                 file_name_suffix=file_name_suffix)


# read back TFRecord gzip file to confirm it was written correctly

tfrecord_infile = '{}-00000-of-00001{}'.format(tfrecord_outfile, file_name_suffix)

def get_raw_dataset(filename):
    filenames = [filename]
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_record(dataset):
    for raw_record in dataset.take(1):
        print(raw_record)
        break
    return raw_record


feature_description = {
    'height': tf.FixedLenFeature([], tf.int64, default_value=0),
    'width': tf.FixedLenFeature([], tf.int64, default_value=0),
    'depth': tf.FixedLenFeature([], tf.int64, default_value=0),
    'label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
}


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, feature_description)


raw_dataset = get_raw_dataset(tfrecord_infile)

raw_record = get_record(raw_dataset)

parsed_dataset = raw_dataset.map(_parse_function)

parsed_record = get_record(parsed_dataset)


def convert_parsed_record_to_ndarray(parsed_record):
    x = parsed_record['image_raw']
    x_np = x.numpy()
    bytestream = io.BytesIO(x_np)
    num_images = 1
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(rows, cols, 1)
    assert isinstance(data, np.ndarray), type(data)
    assert data.shape == (28, 28, 1)
    return data


image_from_infile = convert_parsed_record_to_ndarray(parsed_record)


def display_image(img):
    assert isinstance(img, np.ndarray), type(img)

    stacked_img = np.stack((np.squeeze(img), ) * 3, axis=-1)
    assert stacked_img.shape == (28, 28, 3), stacked_img.shape

    plt.imshow(stacked_img, cmap=plt.get_cmap('gray'))
