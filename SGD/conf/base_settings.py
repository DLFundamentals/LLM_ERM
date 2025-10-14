# device
device = 'cuda'

model = 'gpt2'
sequence_length = 20
context_length = sequence_length + 2
BOS_TOKEN = 2
# batch_size = 1000
# num_batches = 100
# train_set_size = num_batches*batch_size
# test_set_size = batch_size*10
train_set_size = 100000
test_set_size = 10000
batch_size = 1000
num_batches = train_set_size/batch_size

n_epochs = 1000

vocab_size = 3
n_layer = 4
n_head = 10 
n_embd = 500
num_models = 1
ensemble_func = 'mean'

lr = 1e-5
weight_decay = 0.00001
eta_min = 1e-6
online = False

target_func = 'func4'

precision = 'bf16'
num_layers_to_finetune = 2