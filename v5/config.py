import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class myConfig(object):

    ### data file path ###

    # dict_file = 'data/dictionary.txt'
    # glove_file = 'embeddings/new_glove.txt'
    # vocab = 7000+, only train split and set unk for words with freq < 5
    dict_file = '../caption/entity_train_dictionary.txt'
    glove_file = '../embeddings/new_glove_train.txt'

    image_feature_folder = "../Image_Resnet152"
    entity_feature_folder = "../Entity_Resnet152"
    train_caption_path = "../caption/entity_dataset_train_idf.json"
    val_caption_path = "../caption/entity_dataset_val_idf.json"

    g_model_path = ""#"../models/third_train_fixed/g_after_supervised.pth" 
    d_model_path = "" #../models/third_train_fixed/d_after_supervised.pth" 
    

    ### dataloader ###
    supervised_batch_size = 128 # for teacher forcing training
    discriminator_batch_size = 128 # for discriminator training
    

    ### opt_setting ##
    is_train = False
    use_glove = False
    checkpoint_output = None

    ### vocabulary ###
    # will be set main.py, read dictionary and modify the idx and vocab_size
    pad_idx = 0
    start_idx = 0
    end_idx = 0
    vocab_size = 0  # len(dictionary) + <pad> + <\s> + <\e>
    embed_size = 256
    i2w = None
    w2i = None
    
    
    image_hidden = 2048 #25088
    
    ### Generator LSTM model ###
    g_hidden_size = 128 # this is the same for generator

    ### Discriminator LSTM model ###
    d_hidden_size = 128
    lstm_num_layers = 1
    dropout = 0.2

    ### caption process ###
    max_sequence = 15

    ### training epochs ###
    g_train_epoch = 50
    d_train_epoch = 50
    rl_epoch = 1000 # for each rl epoch, we train rl for n_g_rl_epoch and discriminator once
    num_rl_per_epoch = 1
    num_d_per_epoch = 1
    rollout_num = 4

    ### loss and optim ###
    g_crit = torch.nn.CrossEntropyLoss(reduction='none')    # easier to process for mask
    d_crit = torch.nn.BCELoss()
    g_lr = 0.0002 # 0.001
    d_lr = 0.0002

    ### Attention part ###
    # use the same dropout prob
    num_heads = 8
    

    # others
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    def set_optim(self, g_model, d_model):
        self.g_optim = optim.Adam(g_model.parameters(), self.g_lr)
        self.d_optim = optim.Adam(d_model.parameters(), self.d_lr)
        self.g_scheduler = StepLR(self.g_optim, step_size=5, gamma=0.8)
        self.d_scheduler = StepLR(self.d_optim, step_size=5, gamma=0.8)
