import os
import torch
import sys
import getopt
import config
import generator
import discriminator
import train
import utils
import loaders
import infer

def parse_opt(conf):
    opts, args = getopt.getopt(sys.argv[1:], 'e:o:', ['embed=', 'checkpoint_out='])
    for opt_name, opt_value in opts:
        if opt_name in ['-e', '--embeddings']:
            if opt_value == 'glove':
                conf.use_glove = True
            else:
                conf.use_glove = False
        if opt_name in ['-o', '--checkpoint_out']:
            conf.checkpoint_output = opt_value
            



def get_dataloader(conf):
    image_feature_folder = conf.image_feature_folder
    entity_feature_folder = conf.entity_feature_folder
    train_caption_path = conf.train_caption_path
    val_caption_path = conf.val_caption_path
   


    train_supervised_data = loaders.EntityAttnSupervisedDataset(conf, image_feature_folder, entity_feature_folder, train_caption_path)
    train_supervised_loader = loaders.DataLoader(train_supervised_data, batch_size=conf.supervised_batch_size, num_workers=8, shuffle=True, collate_fn=lambda batch: loaders.entity_attn_supervised_collate_fn(batch, conf))
    
    train_discriminator_data = loaders.EntityAttnDiscriminatorDataset(conf, image_feature_folder, entity_feature_folder, train_caption_path)
    train_discriminator_loader = loaders.DataLoader(train_discriminator_data, batch_size=conf.discriminator_batch_size, num_workers=8, shuffle=True, collate_fn=lambda batch: loaders.entity_attn_discriminator_collate_fn(batch, conf))
    
#     val_supervised_data = loaders.EntityAttnSupervisedDataset(conf, image_feature_folder, entity_feature_folder, val_caption_path)
#     val_supervised_loader = loaders.DataLoader(val_supervised_data, batch_size=conf.supervised_batch_size, num_workers=8, shuffle=True, collate_fn=lambda batch: loaders.entity_attn_supervised_collate_fn(batch, conf))
    
#     val_discriminator_data = loaders.EntityAttnDiscriminatorDataset(conf, image_feature_folder, entity_feature_folder, val_caption_path)
#     val_discriminator_loader = loaders.DataLoader(val_discriminator_data, batch_size=conf.discriminator_batch_size, num_workers=8, shuffle=True, collate_fn=lambda batch: loaders.entity_attn_discriminator_collate_fn(batch, conf))

    return train_supervised_loader, train_discriminator_loader, None, None

def main():
    # Initialize the config and model
    conf = config.myConfig()
    parse_opt(conf)

    # read pretrained word embeddings
    if conf.use_glove:
        word_embed, w2i, i2w, conf.pad_idx, conf.start_idx, conf.end_idx, conf.vocab_size =\
            utils.read_new_glove(conf.dict_file, conf.glove_file)
        conf.embed_size = 300
    else:
        word_embed = None
        w2i, i2w, conf.pad_idx, conf.start_idx, conf.end_idx, conf.vocab_size = utils.read_dict(conf.dict_file)
   
    os.makedirs(conf.checkpoint_output, exist_ok=True)
    
    g_model = generator.generator(conf, word_embed)
    d_model = discriminator.discriminator(conf, word_embed).to(conf.device)

    # load model
    if conf.g_model_path != '' and os.path.exists(conf.g_model_path):
        print("loaded G", flush = True)
        g_model.load_state_dict(torch.load(conf.g_model_path, map_location='cpu'))
    if conf.d_model_path != '' and os.path.exists(conf.d_model_path):
        print("loaded D", flush = True)
        d_model.load_state_dict(torch.load(conf.d_model_path, map_location='cpu'))

    # set optimizers
    conf.set_optim(g_model, d_model)
    
    g_model = g_model.to(conf.device)
    d_model = d_model.to(conf.device)
    print("Using device", conf.device, flush=True)
    
    # set up dataloader
    train_supervised_loader, train_discriminator_loader, eval_supervised_loader, eval_discriminator_loader = get_dataloader(conf)
    
    # train model
    train.train(conf, g_model, d_model, train_supervised_loader, train_discriminator_loader, eval_supervised_loader, eval_discriminator_loader)

if __name__ == '__main__':
    main()