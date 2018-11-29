import string

cfg = {}
cfg['char_list'] = string.ascii_lowercase + string.digits + '!$\'()*,-.:;? '
cfg['input_dim'] = 0 # input dimension to LSTM; to be determined after preprocessing the dataset
cfg['hidden_dim'] = 10 # hidden dimension for LSTM
cfg['output_dim'] = len(cfg['char_list']) + 3 # output dimension of the model; + 3 for <SOS>, <EOS>, and <PAD>
cfg['layers'] = 2 # number of layers of LSTM
cfg['dropout'] = 0.05 # dropout rate between two layers of LSTM; useful only when layers > 1; between 0 and 1
cfg['bidirectional'] = False # True or False; True means using a bidirectional LSTM
cfg['batch_size'] = 32 # batch size of input
cfg['learning_rate'] = 0.001 # learning rate to be used
cfg['L2_penalty'] = 0 # weighting constant for L2 regularization term; this is a parameter when you define optimizer
cfg['gen_temp'] = 1 # temperature to use while generating reviews
cfg['max_len'] = 1000 # maximum character length of the generated reviews
cfg['epochs'] = 5 # number of epochs for which the model is trained
cfg['cuda'] = True #True or False depending whether you want to run your model on a GPU or not. If you set this to True, make sure to start a GPU pod on ieng6 server
cfg['train'] = True # True or False; True denotes that the model is bein deployed in training mode, False means the model is not being used to generate reviews