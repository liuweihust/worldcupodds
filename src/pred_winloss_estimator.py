from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from datasets import SoccerDb
from nets import nets_factory

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--model', default='dnn', type=str, help='model:dnn,handmade,default:dnn')
parser.add_argument('--model_dir', default='/tmp/train/', type=str, help='path to save train model')

def main(argv):
    args = parser.parse_args(argv[1:])

    net_fn = nets_factory.get_network(args.model)
    classifier = net_fn(features=SoccerDb.CSV_INPUT_COLUMN_NAMES,model_dir=args.model_dir)

    # Generate predictions from the model
    expected = [ 'HostWin','Deuce','GuestWin']
    predict_x = {
        'HostWin':[1.95],
        'Deuce':[3.57],
        'GuestWin':[4.31]
    }

    predictions = classifier.predict(
        input_fn=lambda:SoccerDb.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SoccerDb.RESULT_NAMES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
