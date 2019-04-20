from __future__ import absolute_import, division

import gzip
import io
import os

import apache_beam as beam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
from logzero import logger
from tensorflow.python.platform import gfile

tf.enable_eager_execution()

TFRECORD_OUTFILE = 'mnist'

FEATURE_DESCRIPTION = {
    'height': tf.FixedLenFeature([], tf.int64, default_value=0),
    'width': tf.FixedLenFeature([], tf.int64, default_value=0),
    'depth': tf.FixedLenFeature([], tf.int64, default_value=0),
    'label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
}


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
    logger.info('Extracting: %s', f.name)
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
    logger.info('Extracting: %s', f.name)
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

    logger.info('images shape: %s', images.shape)
    logger.info('labels shape: %s', labels.shape)

    return images, labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_images_and_labels_w_index(images, labels):
    """
    Attach indexes to each record to be used for `beam.CoGroupByKey`

    TODO: also to be used by `beam._partition_fn` in the future
    """
    images_w_index = [(i, x) for i, x in enumerate(images)]
    labels_w_index = [(i, x) for i, x in enumerate(labels)]
    return images_w_index, labels_w_index


def group_by_tf_example(key_value):
    _, value = key_value
    image = value['image'][0]
    label = value['label'][0]
    height, width, depth = image.shape
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image.tostring())
        }))
    return example

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ImageToExample(pipeline, input_dict):
    data_dir = input_dict['input-base']

    images_path = os.path.join(data_dir, 'images.gz')
    labels_path = os.path.join(data_dir, 'labels.gz')

    images, labels = get_images_and_labels(images_path, labels_path)

    images_w_index, labels_w_index = get_images_and_labels_w_index(
        images, labels)

    # Beam Pipeline
    image_line = pipeline | "CreateImage" >> beam.Create(images_w_index[:1])
    label_line = pipeline | "CreateLabel" >> beam.Create(labels_w_index[:1])
    group_by = ({
        'label': label_line,
        'image': image_line
    }) | beam.CoGroupByKey()
    return (group_by | "GroupByToTfExample" >> beam.Map(group_by_tf_example))


def write_tfrecords():
    """
    Main write function
    """
    input_dict = {'input-base': '/tmp/data/mnist/val/'}
    with beam.Pipeline(options=PipelineOptions()) as p:
        tf_example = p | "InputSourceToExample" >> _ImageToExample(input_dict)

        serialize = (
            tf_example | 'SerializeDeterministically' >>
            beam.Map(lambda x: x.SerializeToString(deterministic=True)))

        (serialize
         | beam.io.WriteToTFRecord(TFRECORD_OUTFILE, file_name_suffix='.gz'))


def get_raw_dataset(filename):
    filenames = [filename]
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_record(dataset):
    return next(iter(dataset.take(1)))


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, FEATURE_DESCRIPTION)


def convert_parsed_record_to_ndarray(parsed_record):
    x = parsed_record['image_raw']
    x_np = x.numpy()
    bytestream = io.BytesIO(x_np)
    rows = 28
    cols = 28
    num_images = 1
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    shape = (rows, cols, num_images)
    data = data.reshape(*shape)
    assert isinstance(data, np.ndarray), type(data)
    assert data.shape == shape
    return data


def read_tfrecord():
    """
    Main read function

    Reads a single image TFRecord and returns it as a np.ndarray
    """
    tfrecord_infile = '{}-00000-of-00001.gz'.format(TFRECORD_OUTFILE)

    raw_dataset = get_raw_dataset(tfrecord_infile)

    raw_record = get_record(raw_dataset)

    parsed_dataset = raw_dataset.map(_parse_function)

    parsed_record = get_record(parsed_dataset)

    return convert_parsed_record_to_ndarray(parsed_record)


def display_image(img):
    """
    Coverts a 1-channel np.ndarray to 3-channel and displays it
    using matplotlib
    """
    assert isinstance(img, np.ndarray), type(img)

    stacked_img = np.stack((np.squeeze(img), ) * 3, axis=-1)
    assert stacked_img.shape == (28, 28, 3), stacked_img.shape

    plt.imshow(stacked_img, cmap=plt.get_cmap('gray'))
