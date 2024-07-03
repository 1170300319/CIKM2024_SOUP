import torch
import torch.nn.functional as F

from models.modeling_q5 import Q5

class Q5PRO(Q5):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')

    def train_step(self, batch):

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        time_ids = batch['time_ids'].to(device)

        pos_labels = batch["query_trigger_ids"].to(device)
        neg_labels = batch["query_trigger_neg_ids"].to(device)

        pos_reward = batch["reward_pos"].to(device)
        neg_reward = batch["reward_neg"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        ## 计算生成概率
        output_pos = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            time_ids=time_ids,
            decoder_input_ids=pos_labels,
            return_dict=True
        )

        prob_pos = output_pos.logits[..., :-1, :].contiguous()
        prob_pos = F.log_softmax(prob_pos, dim=-1)
        prob_pos = torch.gather(prob_pos, 2, pos_labels[:, 1:].unsqueeze(-1)).squeeze(-1)  #[batch, seq_len-1]
        sum_prob_pos = torch.sum(prob_pos, dim=-1) / prob_pos.shape[-1]  # 平均概率

        output_neg = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            time_ids=time_ids,
            decoder_input_ids=neg_labels,
            return_dict=True
        )

        prob_neg = output_neg.logits[..., :-1, :].contiguous()
        prob_neg = F.log_softmax(prob_neg, dim=-1)
        prob_neg = torch.gather(prob_neg, 2, neg_labels[:, 1:].unsqueeze(-1)).squeeze(-1)  #[batch, seq_len-1]
        sum_prob_neg = torch.sum(prob_neg, dim=-1) / prob_neg.shape[-1]  # 平均概率

        # PRO
        eps = 1e-10
        neg_temperatures = pos_reward.view(-1, 1) - neg_reward  # [batch, training_stage-time-1]
        pos_temperature = torch.max(neg_temperatures, dim=1).values  # [batch]
        pro_loss = torch.log(eps + torch.exp(sum_prob_pos * pos_temperature) + torch.sum(
            torch.exp(sum_prob_neg * neg_temperatures), dim=1)) - sum_prob_pos * pos_temperature  # [batch]
        pro_loss = torch.mean(pro_loss).to(output_pos.decoder_last_hidden_state.dtype)

        # SFT
        sft_weight = 0.5
        sft_loss = torch.mean(-sum_prob_pos).to(output_pos.decoder_last_hidden_state.dtype)
        sft_loss = sft_weight * sft_loss

        loss = sft_loss + pro_loss
        # output = self(
        #     input_ids=input_ids,
        #     whole_word_ids=whole_word_ids,
        #     time_ids=time_ids,
        #     labels=lm_labels,
        #     return_dict=True
        # )
        # assert 'loss' in output
        # lm_mask = lm_labels != -100 # true or false
        # lm_mask = lm_mask.float() # 1 or 0 ？
        # B, L = lm_labels.size() # labels中满足给定条件的大小
        # loss = output['loss']
        # 这一步的作用是什么
        # loss = loss.view(B, L) * lm_mask
        # loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = 1

        # task_counts = {task: 0 for task in self.losses}
        # task_loss = {task: 0 for task in self.losses}
        # for _loss, task in zip(loss.detach(), batch['task']):
        #     task_loss[task] += _loss
        #     task_counts[task] += 1
        # for task in self.losses:
        #     if task_counts[task] > 0:
        #         results[f'{task}_loss'] = task_loss[task]
        #         results[f'{task}_loss_count'] = task_counts[task]

        return results

    # 测试在正样本上的结果
    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)

        lm_labels = batch["query_trigger_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = 1
        # results['total_loss_count'] = len(loss)

        # task_counts = {task: 0 for task in self.losses}
        # task_loss = {task: 0 for task in self.losses}
        #
        # for _loss, task in zip(loss.detach(), batch['task']):
        #     task_loss[task] += _loss
        #     task_counts[task] += 1

        # for task in self.losses:
        #     if task_counts[task] > 0:
        #         results[f'{task}_loss'] = task_loss[task]
        #         results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents
