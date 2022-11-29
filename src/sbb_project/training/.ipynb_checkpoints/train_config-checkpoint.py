class Hparams:
    def __init__(self):
        self.epochs = 10  # number of training epochs
        self.seed = 42  # randomness seed
        self.cuda = True  # use nvidia gpu
        self.save = "./models/"  # save checkpoint


def get_train_config():
    return Hparams()