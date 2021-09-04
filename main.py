import argparse
from model import train, test, MenN2N
import os
from data import read_data, load_vocab
import paddle

parser = argparse.ArgumentParser()
parser.add_argument("--edim", default=150, type=int,
                    help="internal state dimension [150]")
parser.add_argument("--lindim", default=75, type=int,
                    help="linear part of the state [75]")
parser.add_argument("--nhop", default=6, type=int, help="number of hops [6]")
parser.add_argument("--mem_size", default=100, type=int,
                    help="memory size [100]")
parser.add_argument("--batch_size", default=128, type=int,
                    help="batch size to use during training [128]")
parser.add_argument("--nepoch", default=100, type=int,
                    help="number of epoch to use during training [100]")
parser.add_argument("--init_lr", default=0.01, type=float,
                    help="initial learning rate [0.01]")
parser.add_argument("--init_hid", default=0.1, type=float,
                    help="initial internal state value [0.1]")
parser.add_argument("--init_std", default=0.05, type=float,
                    help="weight initialization std [0.05]")
parser.add_argument("--max_grad_norm", default=50, type=int,
                    help="clip gradients to this norm [50]")
parser.add_argument("--data_dir", default="data", type=str,
                    help="data directory [data]")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                    help="checkpoint directory [checkpoints]")
parser.add_argument("--model_name", default="model", type=str,
                    help="model name for test [model]")
parser.add_argument("--recover_train", default=False, type=bool,
                    help="if True, load model [model_name] before train [False]")
parser.add_argument("--data_name", default="ptb", type=str,
                    help="data set name [ptb]")
parser.add_argument("--is_test", default=False, type=bool,
                    help="True for testing, False for Training [False]")
parser.add_argument("--show", default=False, type=bool,
                    help="print progress, need progress module [False]")
config = parser.parse_args()

if __name__ == '__main__':
    paddle.set_device("gpu")

    vocab_path = os.path.join(config.data_dir,
                              "%s.vocab.txt" % config.data_name)
    word2idx = load_vocab(vocab_path)

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    train_data = read_data(
        os.path.join(config.data_dir, "%s.train.txt" % config.data_name),
        word2idx)
    valid_data = read_data(
        os.path.join(config.data_dir, "%s.valid.txt" % config.data_name),
        word2idx)
    test_data = read_data(
        os.path.join(config.data_dir, "%s.test.txt" % config.data_name),
        word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    config.nwords = len(word2idx)

    print("vacab size is %d" % config.nwords)

    model = MenN2N(config)
    if not config.is_test:
        if config.recover_train:
            model_path = os.path.join(config.checkpoint_dir, config.model_name)
            state_dict = paddle.load(model_path)
            model.set_dict(state_dict)
        train(model, train_data, valid_data, config)
    else:
        model_path = os.path.join(config.checkpoint_dir, config.model_name)
        state_dict = paddle.load(model_path)
        model.set_dict(state_dict)
        test(model, test_data, config)
