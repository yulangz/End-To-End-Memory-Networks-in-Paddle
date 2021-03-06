class Empty():
    pass

config = Empty()

config.edim = 150                       # internal state dimension
config.lindim = 75                      # linear part of the state
config.nhop = 7                         # number of hops
config.mem_size = 200                   # memory size
config.batch_size = 128                 # batch size to use during training
config.nepoch = 100                     # number of epoch to use during training
config.init_lr = 0.01                   # initial learning rate
config.init_hid = 0.1                   # initial internal state value
config.init_std = 0.05                  # weight initialization std
config.max_grad_norm = 50               # clip gradients to this norm
config.data_dir = "data"                # data directory
config.checkpoint_dir = "checkpoints"   # checkpoint directory
config.model_name = "model"             # model name for test and recover train
config.recover_train = False            # if True, load model [model_name] before train
config.data_name = "ptb"                # data set name
config.show = True                      # print progress, need progress module
config.srand = 17814                    # initial random seed
