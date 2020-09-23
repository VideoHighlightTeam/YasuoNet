from data_loader import DataLoader
from model import *
from trainer import Trainer
from tensorflow.keras.optimizers import Adam
import argparse

parser = argparse.ArgumentParser(description="Train YasuoNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default='train', type=str, dest='mode', required=True)
parser.add_argument("--data_dir", type=str, dest='data_dir', required=True)
parser.add_argument("--batch_size", type=int, dest='batch_size', required=True)
parser.add_argument("--epochs", type=int, dest='epochs', required=True)
parser.add_argument("--learning_rate", default=1e-3, type=float, dest='learning_rate')
parser.add_argument("--ckpt_dir", default='./checkpoints', type=str, dest='ckpt_dir')
# parser.add_argument("--train_continue", default='off', type=str, dest='train_continue')

args = parser.parse_args()

# parameter
mode = args.mode
data_dir = args.data_dir
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
ckpt_dir = args.ckpt_dir
# train_continue = args.train_continue


def main():
    # for basic model
    # data_loader = DataLoader(data_dir, x_includes=['video', 'audio'])
    # input_shape_dict = data_loader.get_metadata()['data_shape']
    # model = build_basic_model(input_shape_dict)

    # for sequence model
    data_loader = DataLoader(data_dir, x_includes=['video', 'audio'], x_expand=2)
    input_shape_dict = data_loader.get_metadata()['data_shape']
    model = build_sequence_model(input_shape_dict)

    if mode == 'train':
        model.summary()

        class_weights = (1, 1)

        trainer = Trainer(model, data_loader, ckpt_dir)
        trainer.train(Adam(learning_rate), epochs, batch_size, class_weights)
    elif mode == 'test':
        trainer = Trainer(model, data_loader, ckpt_dir)
        trainer.test()
    elif mode == 'predict':
        pass


if __name__ == '__main__':
    main()
