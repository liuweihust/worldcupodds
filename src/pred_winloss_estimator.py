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

    (pred_x,hosts,guests) = SoccerDb.load_preddata()
    net_fn = nets_factory.get_network(args.model)
    classifier = net_fn(features=pred_x.keys(),model_dir=args.model_dir)

    # Generate predictions from the model
    predictions = classifier.predict(
        input_fn=lambda:SoccerDb.eval_input_fn(pred_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    SoccerDb.PrintDict()
    template = ('\n"{}":"{}" Prediction is "{}" ({:.1f}% {:.1f}% {:.1f}%)')

    for n1,n2,pred_dict in zip(hosts,guests,predictions):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(n1,n2,SoccerDb.RESULT_NAMES[class_id],
                              100 * pred_dict['probabilities'][0],
                              100 * pred_dict['probabilities'][1],
                              100 * pred_dict['probabilities'][2])
                            )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
