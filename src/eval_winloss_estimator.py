from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from datasets import SoccerDb
from nets import nets_factory

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--model', default='dnn', type=str, help='model:dnn,handmade,default:dnn')
parser.add_argument('--train_steps', default=60000, type=int,
                    help='number of training steps')
parser.add_argument('--model_dir', default='../train/', type=str, help='path to save train model')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (eval_x, eval_y) = SoccerDb.load_traindata()

    net_fn = nets_factory.get_network(args.model)
    classifier = net_fn(features=eval_x.keys(),model_dir=args.model_dir)

    # Train the Model.
    eval_result = classifier.evaluate(
        input_fn=lambda:SoccerDb.eval_input_fn(eval_x, eval_y,
                                                 args.batch_size),
        )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
