#!/bin/sh -x

Usage()
{
	echo "Usage:"
	echo "./run.sh train|pred 310|pts [model_path,default:/tmp/train/]"
}

mode='train'
#mode='pred'
data='310'
modelpath='/tmp/train310/'
learning_rate=0.01
train_steps=20000

if [ $# -ge 1 ]; then
	mode=$1
fi

if [ $# -ge 2 ]; then
	data=$2
fi

if [ $# -ge 3 ]; then
	modelpath=$3
fi

echo "Will do:$mode with $data at $modelpath learning_rate=$learning_rate steps=$train_steps"

case "$data" in
	"310")
		net_param="--num_layer 4 --num_size 128"
		;;
	"pts")
		net_param="--num_layer 4 --num_size 256"
		;;
	*)
		echo "Error mode"
		;;
esac

case "$mode" in
	"train")
		python train_winloss_estimator.py --dataset $data --model_dir $modelpath	\
			$net_param --learning_rate=$learning_rate --train_steps=$train_steps	\
			--dropout 0.3 								\
			--batch_size 64								\
			#--activation_fn sigmoid
		;;
	"pred")
		python pred_winloss_estimator.py --dataset $data --model_dir $modelpath		\
		$net_param
		;;
	*)
		echo "Error mode"
		;;
esac
