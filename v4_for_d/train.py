from config import myConfig
from generator import generator
from discriminator import discriminator
from rollout import train_gd_with_rl
import utils
import time
import torch
from os.path import join as P_join
from utils import decode_sentence

def train_g(config, g_model, train_dataloader, eval_dataloader, num_epoch, caller="supervised"):
    """
    Train g model alone, use teacher forcing to train and no use sampling
    """
    print('begin to train g model alone...')


    for e in range(num_epoch):
        g_model.train()
        print('cur_epoch:', e, flush=True)
        running_loss = 0
        start_time = time.time()
        for idx, data in enumerate(train_dataloader):
            # TODO: the definition of dataloader is not defined
            entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"]
            entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len)

            # pred_word_probs: [batch, seq_len-1, vocab], sampled_sent: [batch, seq_len], sampled_sent_len: [batch]
#             print(target_sent, target_sent_len)
            pred_word_probs, sampled_sent, sampled_sent_len = g_model(entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len)
#             assert (torch.sum(sampled_sent == target_sent) == sampled_sent.shape[0] * sampled_sent.shape[1])

            # compute loss
            # first to convert pred_word_probs, sampled_sent to 2D [batch_size * (seq_len-1), vocab], [batch * (seq-1)]
            # then use crossentropy(reduce=None) -> [batch_size * (seq-1)]
            # then mask and compute mean
            flat_pred_word_probs = pred_word_probs.reshape(-1, pred_word_probs.size(2))
            flat_sampled_sent = sampled_sent[:, 1:].reshape(-1)                 # ignore START
            loss = config.g_crit(flat_pred_word_probs, flat_sampled_sent)       # [batch_size * (seq_len - 1)]
            mask = (flat_sampled_sent != config.pad_idx).type(torch.FloatTensor).to(config.device)
#             mask3 = (sampled_sent != config.pad_idx).to(config.device)
            
#             mask2 = torch.arange(target_sent.shape[1]).to(config.device).expand(target_sent.shape[0], target_sent.shape[1]) < (target_sent_len)
            
#             assert (torch.sum(mask3 == mask2) == mask2.shape[0] * mask2.shape[1])
            
            loss = torch.sum(loss * mask) / torch.sum(mask)                              # compute mean

            running_loss += loss.item()
            config.g_optim.zero_grad()
            loss.backward()
            config.g_optim.step()
            config.g_optim.zero_grad()
        end_time = time.time()
        running_loss /= len(train_dataloader)
        print('G Training Loss:', running_loss, 'Time:', end_time - start_time, 's', flush=True)
        
        if (e+1)%10 == 0 and eval_dataloader is not None:
            eval_g(config, g_model, eval_dataloader)
            torch.save(g_model.state_dict(), P_join(config.checkpoint_output, "g_"+caller+"_epoch"+str(e)+".pth"))

def train_d(config, g_model, d_model, train_dataloader, eval_dataloader, num_epoch, caller="supervised"):
    """
    Train d model alone.
    """
    print('begin to train d model alone...', flush=True)
    for e in range(num_epoch):
        g_model.eval()              # for generate sentence
        d_model.train()
        print('cur_epoch:', e)
        running_loss = 0
        loss1 = 0
        loss2 = 0
        loss3 = 0
        first_score = 0
        start_time = time.time()
        for idx, data in enumerate(train_dataloader):
            # TODO: the definition of dataloader is not defined
            # TODO: in d model training, we need 3 examples: target / generated / random sample from other imgs
            # Here is ground truth loss
            entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"], data["wrong_caption"], data["wrong_length"]
            entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len)
            batch_size = img_feature.size(0)
            target_label = torch.ones(batch_size).to(config.device)
            pred_scores = d_model(entity_feature, target_sent, target_sent_len)
            
#             if e == 2:
#                 print('pred_scores:', pred_scores)
            temp = config.d_crit(pred_scores, target_label)
            loss = temp
            loss1 += temp.item()

            # Here is generated sent loss
            # pred_word_probs: [batch, seq_len-1, vocab], sampled_sent: [batch, seq_len], sampled_sent_len: [batch]
            with torch.no_grad():  # no grads since we do not compute grad in g model
                pred_word_probs, sampled_sent, sampled_sent_len = g_model(entity_feature, neighbor_list, img_feature)
                pred_word_probs, sampled_sent, sampled_sent_len = pred_word_probs.detach(), sampled_sent.detach(), sampled_sent_len.detach()
            ge_pred_scores = d_model(entity_feature, sampled_sent, sampled_sent_len)
            if idx == 0:
                first_score = torch.mean(ge_pred_scores).item()
            ge_target_label = torch.zeros(batch_size).to(config.device)
#             if e == 2:
#                 print('ge_pred_scores:', ge_pred_scores)
            temp = config.d_crit(ge_pred_scores, ge_target_label)
            loss += temp
            loss2 += temp.item()
            

            # Here is random sampled loss
            wrong_target_label = torch.zeros(batch_size).to(config.device)
            wrong_pred_scores = d_model(entity_feature, wrong_sent, wrong_sent_len)
#             if e == 2:
#                 print('wrong_pred_scores:', wrong_pred_scores)
            temp = config.d_crit(wrong_pred_scores, wrong_target_label)
            loss += temp
            loss3 += temp.item()
            
            running_loss += loss.item()
            config.d_optim.zero_grad()
            loss.backward()
            config.d_optim.step()
            config.d_optim.zero_grad()
            
#             temp1 = target_sent.cpu().detach().numpy()
#             temp2 = wrong_sent.cpu().detach().numpy()
#             temp3 = sampled_sent.cpu().detach().numpy()
#             target_list = decode_sentence(temp1, i2w)
#             wrong_list = decode_sentence(temp2, i2w)
#             gen_list = decode_sentence(temp3, i2w)
#             print("_____________new___________")
#             print(target_list[0])
#             print(wrong_list[0])
#             print(gen_list[0])
#             print(data['caption_name'][0])
            
            
        end_time = time.time()
        running_loss /= len(train_dataloader)
        loss1 /= len(train_dataloader)
        loss2 /= len(train_dataloader)
        loss3 /= len(train_dataloader)
        print('D Training Loss:', running_loss, 'Time:', end_time - start_time, 's', flush=True)
        print("loss of true", loss1, "loss of gen", loss2, "loss of other", loss3, "first score", first_score, flush=True)
        
        if (e+1)%10 == 0 and eval_dataloader is not None:
            eval_d(config, g_model, d_model, eval_dataloader)
            torch.save(g_model.state_dict(), P_join(config.checkpoint_output, "d_"+caller+"_epoch"+str(e)+".pth"))

def eval_g(config, g_model, eval_dataloader):
    """
    similar with train_g, use teacher forcing and compute loss
    """
    print('begin to eval g model...')
    g_model.eval()
    g_model.to(config.device)
    with torch.no_grad():
        running_loss = 0
        start_time = time.time()
        for batch_idx, data in enumerate(eval_dataloader):
            # TODO: the definition of dataloader is not defined
            entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"]
            entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len)

            # pred_word_probs: [batch, seq_len-1, vocab], sampled_sent: [batch, seq_len], sampled_sent_len: [batch]
            pred_word_probs, sampled_sent, sampled_sent_len = g_model(entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len)

            # compute loss
            # first to convert pred_word_probs, sampled_sent to 2D [batch_size * (seq_len-1), vocab], [batch * (seq-1)]
            # then use crossentropy(reduce=None) -> [batch_size * (seq-1)]
            # then mask and compute mean
            flat_pred_word_probs = pred_word_probs.reshape(-1, pred_word_probs.size(2))
            flat_sampled_sent = sampled_sent[:, 1:].reshape(-1)  # ignore START
            loss = config.g_crit(flat_pred_word_probs, flat_sampled_sent)  # [batch_size * (seq_len - 1)]
            mask = (flat_sampled_sent != config.pad_idx).type(torch.FloatTensor).to(config.device)
            loss = torch.sum(loss * mask) / torch.sum(mask)  # compute mean

            running_loss += loss.item()
        end_time = time.time()
        running_loss /= len(eval_dataloader)
        print('G Eval Loss:', running_loss, 'Time:', end_time - start_time, 's', flush=True)

def eval_d(config, g_model, d_model, eval_dataloader):
    """
    similar with train_d, eval in three types of data
    groundtruth / generated sent / random sampled
    """
    print('begin to eval d model alone...')
    g_model.eval()  # for generate sentence
    g_model.to(config.device)
    d_model.eval()
    d_model.to(config.device)

    with torch.no_grad():
        running_loss = 0
        loss1 = 0
        loss2 = 0
        start_time = time.time()
        for idx, data in enumerate(eval_dataloader):
            # TODO: the definition of dataloader is not defined
            # TODO: in d model training, we need 3 examples: target / generated / random sample from other imgs
            # Here is ground truth loss
            entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"], data["wrong_caption"], data["wrong_length"]
            entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_list, img_feature, target_sent, target_sent_len, wrong_sent, wrong_sent_len)
            
            batch_size = img_feature.size(0)
            target_label = torch.zeros(batch_size).long().to(config.device)
            pred_scores = d_model(entity_feature, target_sent, target_sent_len)
#             print('pred_scores:', pred_scores)
            temp = config.d_crit(pred_scores, target_label)
            loss = temp
            loss1 += temp.item()
            # Here is generated sent loss
            # pred_word_probs: [batch, seq_len-1, vocab], sampled_sent: [batch, seq_len], sampled_sent_len: [batch]
#             pred_word_probs, sampled_sent, sampled_sent_len = g_model(img_feature)
#             ge_pred_scores = d_model(entity_feature, sampled_sent, sampled_sent_len)
#             ge_target_label = torch.zeros(batch_size).to(config.device)
#             loss += config.d_crit(ge_pred_scores, ge_target_label)

            # Here is random sampled loss
            wrong_target_label = torch.zeros(batch_size).fill_(2).long().to(config.device)
            wrong_pred_scores = d_model(entity_feature, wrong_sent, wrong_sent_len)
#             print('wrong_pred_scores:', wrong_pred_scores)
            temp = config.d_crit(wrong_pred_scores, wrong_target_label)
            loss += temp
            loss2 += temp.item()

            running_loss += loss.item()
        end_time = time.time()
        running_loss /= len(eval_dataloader)
        loss1 /= len(eval_dataloader)
        loss2 /= len(eval_dataloader)
        print('D Eval Loss:', running_loss, 'Time:', end_time - start_time, 's', flush=True)
        print('D Eval: loss of true', loss1, 'loss of other', loss2, flush=True)

def train(config, g_model, d_model, train_supervised_loader, train_discriminator_loader, eval_supervised_loader=None, eval_discriminator_loader = None):
    """
    The models should already be moved to the device.
    In the training phase:
    1. train g model alone and eval
    2. train d model alone and eval
    3. train g+d model with reinforce learning
    """

    train_g(config, g_model, train_supervised_loader, eval_supervised_loader, config.g_train_epoch)
    torch.save(g_model.state_dict(), P_join(config.checkpoint_output, "g_after_supervised.pth"))

#     train_d(config, g_model, d_model, train_discriminator_loader, eval_discriminator_loader, config.d_train_epoch)
#     torch.save(d_model.state_dict(), P_join(config.checkpoint_output, "d_after_supervised.pth"))

    config.set_optim(g_model, d_model)
        
    train_gd_with_rl(config, g_model, d_model, train_supervised_loader, train_discriminator_loader, eval_supervised_loader, eval_discriminator_loader, config.rl_epoch, config.num_rl_per_epoch, config.num_d_per_epoch)
    torch.save(g_model.state_dict(), P_join(config.checkpoint_output, "g_after_rl.pth"))
    torch.save(d_model.state_dict(), P_join(config.checkpoint_output, "d_after_rl.pth"))


if __name__ == '__main__':
    config = myConfig()
    g_model = generator(config)
    img_feature = torch.randn(2, config.image_hidden)
    target_sent = torch.LongTensor([[1, 1, 0], [1, 0, 1]])
    target_sent_len = torch.LongTensor([3, 3])
    fix_seq_len = -1
    g_model(img_feature, target_sent, target_sent_len, fix_seq_len)
