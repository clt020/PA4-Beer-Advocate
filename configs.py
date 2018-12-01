import string

cfg = {}

# string containing all the valid characters in the review text
cfg['valid_char'] = string.ascii_lowercase + string.digits + '!$\'()*,-.:;? '
cfg['valid_char_len'] = len(cfg['valid_char'])

# number of data points to use
cfg['num_data'] = 5000

# parameters of the models (will also be updated in code)
cfg['input_dim'] = 0 # input dimension of the model; will be updated after processing the dataset
cfg['hidden_dim'] = 32 # hidden dimension of the model
cfg['output_dim'] = cfg['valid_char_len'] + 3 # output dimension of the model; + 3 for <SOS>, <EOS>, and <PAD>
cfg['layers'] = 1 # number of hidden layers in the model

# parameters for training
cfg['train_percentage'] = 0.8
cfg['epochs'] = 15
cfg['batch_size'] = 32
cfg['learning_rate'] = 0.01
cfg['early_stop'] = 3

# parameters for review generation
cfg['gen_temp'] = 0.5 # temperature to use while generating reviews
cfg['max_len'] = 1000 # maximum character length of the generated reviews




# not used (might delete after finalizing code)
cfg['cuda'] = True #True or False depending whether you want to run your model on a GPU or not. If you set this to True, make sure to start a GPU pod on ieng6 server
cfg['bidirectional'] = False # True or False; True means using a bidirectional LSTM
cfg['L2_penalty'] = 0 # weighting constant for L2 regularization term; this is a parameter when you define optimizer
cfg['train'] = True # True or False; True denotes that the model is bein deployed in training mode, False means the model is not being used to generate reviews
cfg['dropout'] = 0 # dropout rate between layers in the model; useful only when layers > 1; between 0 and 1