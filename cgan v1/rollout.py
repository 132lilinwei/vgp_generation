import torch
import torch.nn.functional as F
import time
import train
from os.path import join as P_join

def get_rollout_rewards(config, rollout_policy, input_x, input_x_len, img_feature, rollout_num, discriminator):
    """
    output:
        rewards: [batch_size, input_x_len-1], reward for outputing the step t+1's input (the first step is skipped)                        
    """
    rollout_policy.eval()
    with torch.no_grad():
        max_seq_length = input_x.shape[1]
        rewards = torch.zeros((input_x.shape[0], input_x.shape[1]-1)).to(config.device)
        for i in range(rollout_num):
            # rewards for any sample's all t are computed, but when calculating loss , remember to use mask to ignore the ones after END
            # we also ignore the index at 0 since P(a|s) for START is meaningless
            for t in range(1, max_seq_length - 1):
                _, rollout_samples, sample_len = rollout_policy(img_feature = img_feature, target_sent = input_x, target_sent_len = input_x_len, fix_seq_len = t)
                scores = discriminator(img_feature = img_feature, sentence = rollout_samples, sen_seq_len = sample_len)
                rewards[:,t-1] = rewards[:,t-1] + scores  
            scores = discriminator(img_feature = img_feature, sentence = input_x, sen_seq_len = input_x_len)
            rewards[:,max_seq_length-2] = rewards[:,max_seq_length-2] + scores # -2 because we minus one more to discard START
            rewards = rewards.detach()
        rewards = rewards/rollout_num
        print("batch reward last col mean", torch.mean(rewards[:,max_seq_length-2]).item(), "first col mean", torch.mean(rewards[:,0]).item(), "all mean", torch.mean(rewards).item())
    rollout_policy.train()
    return rewards

def compute_rl_loss(config, generator, discriminator, words_probs, input_x, input_x_len, img_feature, rollout_num, input_noise = None):
    rewards = get_rollout_rewards(config, generator, input_x, input_x_len, img_feature, rollout_num, discriminator)
    
    # log(P(a|s))
    # mask to extract corresponding input_x as actions, discard first column because START is not an action
    tmp_len = input_x_len.view(-1)
    max_len = input_x.shape[1]
    mask = torch.arange(max_len).to(config.device).expand(input_x.shape[0], max_len) < tmp_len.unsqueeze(1)
    mask[:,0] = 0
    actions = input_x[mask].view(-1)

    # mask to extract corresponding prob_x as P(a|s)
    mask = torch.zeros(words_probs.shape)
    mask = torch.arange(max_len-1).to(config.device).expand(input_x.shape[0], max_len-1) < (tmp_len-1).unsqueeze(1)
    X_prob = words_probs[mask].view((-1, words_probs.shape[2]))
    clean_rewards = rewards[mask].view(-1) 
    
    loss = F.cross_entropy(X_prob, actions, reduction="none")
    loss = torch.mean(loss * clean_rewards)
    return loss, rewards


def train_gd_with_rl(config, generator, discriminator, train_supervised_loader, train_discriminator_loader, eval_supervised_loader, eval_discriminator_loader, num_epoch, num_rl_per_epoch, num_d_per_epoch, input_noise = None):
    start_time = time.time()
    print("begin rl....", flush=True)
    for e in range(num_epoch):
        print("rl epoch {}, begin RL for generator...".format(e), flush=True)
        generator.train()
        discriminator.eval()
        for i in range(num_rl_per_epoch):
            total_loss = 0
            for ct, data in enumerate(train_supervised_loader):
                img_feature, _, _ = data["image"], data["caption"], data["length"]
                img_feature = img_feature.to(config.device)
                words_probs, sampled_sent,  sampled_sent_len= generator(img_feature = img_feature, target_sent = None, target_sent_len = None, fix_seq_len = None)
                loss, rewards = compute_rl_loss(config, generator, discriminator, words_probs, sampled_sent, sampled_sent_len, img_feature, config.rollout_num, input_noise)
                
                config.g_optim.zero_grad()
                loss.backward()
                for param in generator.parameters():
                    param.grad.data.clamp_(-1, 1)
                config.g_optim.step()
                config.g_optim.zero_grad()
                total_loss += loss.item()
                print("rl training, epoch{}, iter{}, batch{}/{}, batch loss:{}, Training time:{}".format(e, i, ct,len(train_supervised_loader), loss.item(), time.time()-start_time), flush=True)
                
                if torch.mean(rewards).item() > 0.95:
                    print("RL early break", flush=True)
                    break
            total_loss = total_loss / (ct+1)
            print("rl training, epoch {}, iter {}, loss:{}, Training time:{} ".format(e, i, total_loss, time.time()-start_time), flush=True)
        torch.save(generator.state_dict(), P_join(config.checkpoint_output, "gen_after_rl_epoch_"+str(e)+".pth"))

        # training discriminator again
        # generator.eval()
        # discriminator.train()
        
        print("rl epoch {}, begin RL for discriminator...".format(e), flush=True)
        train.train_d(config, generator, discriminator, train_discriminator_loader, eval_discriminator_loader, num_epoch=num_d_per_epoch, caller="rl")
        torch.save(discriminator.state_dict(), P_join(config.checkpoint_output, "discriminator_after_rl_epoch_"+str(e)+".pth"))
      






