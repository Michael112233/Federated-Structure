# about dataset
dataset_name = 'mnist' # ''
sampling_kind = 'iid' # iid and non-iid
training_samples_ratio = 0.8

# about structure
algorithm_name = 'FedAvg_SGD'
client_num = 100
participate_ratio = 0.1
batch_size = 64
global_iter = 1000
local_iter = 50

# about model (logistic, svm, neural)
model_name = 'neural'
hidden_dim = 50

# training
eta_option = 'sqrt'
eta = 0.5
max_grad_time_mul = 300
radius = 1e-4

# non_iid parameter
non_iid_bar = 3

# whether print info
verbose = True
