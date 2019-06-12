from absl import app
from absl import flags
from absl import logging

from onnx2graphml import convert

FLAGS = flags.FLAGS

flags.DEFINE_string('onnx_file', None, 'ONNX model file input')
flags.DEFINE_string('graphml_file', 'model.graphml', 'Destination for GraphML model file output')

flags.mark_flag_as_required('onnx_file')


def main(argv):
  del argv  # Unused.

  logging.info('Running onnx2graphml command-line utility.')
  convert.convert(FLAGS.onnx_file, FLAGS.graphml_file)
  logging.info('%s converted and written to %s successfully.' % (FLAGS.onnx_file, FLAGS.graphml_file))


if __name__ == '__main__':
  app.run(main)
