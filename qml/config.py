import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(
        '---------------------- {0} ----------------------'.format(notifi_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('---------------------- END ----------------------')


def print_config(config):
    content_list = []
    args = list(vars(config))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(config, arg))]
    print_notification(content_list, 'CONFIG')


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")

main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--data_dir", type=str,
                       default="../_data/hymenoptera_data/",
                       help="Directory with hymenoptera data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=4,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=25,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=4,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=4,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--resume", type=str2bool,
                       default=False,
                       help="Whether to resume training from existing checkpoint")
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--loss_type", type=str,
                       default="CrossEntropyLoss",
                       help="Type of data loss to be used")

model_arg.add_argument("--activation", type=str,
                       default="ReLU",
                       help="Activation type")

model_arg.add_argument("--normalize", type=str2bool,
                       default=True,
                       help="Whether to normalize with mean/std or not")

model_arg.add_argument("--l2_reg", type=float,
                       default=1e-4,
                       help="L2 Regularization strength")

model_arg.add_argument("--n_qubits", type=int,
                       default=4,
                       help="Number of qubits in quantum circuit")

model_arg.add_argument("--q_depth", type=int,
                       default=6,
                       help="Depth of the quantum circuit (number of variational layers)")

model_arg.add_argument("--q_delta", type=int,
                       default=0.01,
                       help="Initial spread of random quantum weights")

model_arg.add_argument("--crop_size", type=int,
                       default=224,
                       help="Final image cropping to (size, size)")

model_arg.add_argument("--model", type=str,
                       default="classic",
                       choices=["classic", "quantum"],
                       help="Choose a classical model or a quantum model.")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
