import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import BatchType, ModeType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type, relation_choice, sample):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]
            relation_choice: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_choice = torch.index_select(
                self.choice_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type, relation_choice, sample), (head, tail)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score, _ = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score, ent = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                ent[0].norm(p=2)**2 +
                ent[1].norm(p=2)**2
            ) / ent[0].shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        logs_rel = defaultdict(list)  # logs for every relation

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, _ = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        rel = positive_sample[i][1].item()

                        ranking = 1 + ranking.item()

                        log = {
                            '*************  Model '+args.model + '*****************': 1,
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        }

                        logs.append(log)
                        logs_rel[rel].append(log)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        metrics_rel = defaultdict(dict)
        for rel in logs_rel:
            for metric in logs_rel[rel][0].keys():
                metrics_rel[rel][metric] = sum([log[metric] for log in logs_rel[rel]]) / len(logs_rel[rel])

        return metrics, metrics_rel

    def ReflEc(self, head, rel, tail, batch_type, sel):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = rel/(self.embedding_range.item()/self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_rhead = (re_head * re_relation + im_head * im_relation)
        re_score = re_rhead- re_tail
        im_rhead = (re_head * im_relation - im_head * re_relation)
        im_score = im_rhead - im_tail

        score = torch.stack([re_score*sel, im_score*sel], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item()*0 - score.sum(dim=2)
        return score, re_rhead, im_rhead

    def RotatE(self, head, rel, tail, batch_type, sel):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = rel/(self.embedding_range.item()/self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_rhead = re_head * re_relation - im_head * im_relation
        re_score = re_rhead - re_tail
        im_rhead = re_head * im_relation + im_head * re_relation
        im_score = im_rhead - im_tail

        score = torch.stack([re_score*sel, im_score*sel], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item()*0 - score.sum(dim=2)
        return score, re_rhead, im_rhead

    def TransE(self, head, rel, tail, batch_type, sel):
        rhead = head + rel
        score = rhead - tail
        score = (score*sel).norm(dim=-1, p=2)
        score = self.gamma.item()*0 - score
        re_rhead, im_rhead = torch.chunk(rhead, 2, dim=2)
        return score, re_rhead, im_rhead

    def ScalE(self, head, rel, tail, batch_type, sel):

        rhead = head * torch.concatenate([rel, rel], dim=-1)
        score = rhead - tail

        score = (score*sel).norm(dim=-1, p=2)
        score = self.gamma.item()*0 - score
        re_rhead, im_rhead = torch.chunk(rhead, 2, dim=2)
        return score, re_rhead, im_rhead

    def newModEl(self, head, rel, tail, batch_type, sel):
        ############################
        ### Write your code here ###
        ############################
        score = self.gamma.item()*0 - head*rel*tail
        re_rhead, im_rhead = head, head
        return score, re_rhead, im_rhead

class smartFW(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm,
                 load_choice_embs=False, single_egt=False, threshold=0.0, models_list=['TransE', 'RotatE', 'ReflEc', 'ScalE']):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm
        self.rdimMul = 0
        self.saveRdimMul = []
        self.models_list = models_list
        self.models = []
        for model in models_list:
            if model == 'TransE':
                self.models.append(self.TransE)
                self.rdimMul += 2
            if model == 'RotatE':
                self.models.append(self.RotatE)
                self.rdimMul += 1
            if model == 'ReflEc':
                self.models.append(self.ReflEc)
                self.rdimMul += 1
            if model == 'ScalE':
                 self.models.append(self.ScalE)
                 self.rdimMul += 1
            if model == 'newModEl':
                self.models.append(self.newModEl)
                self.rdimMul += 1
            self.saveRdimMul.append(self.rdimMul)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(self.num_entity, self.hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.num_relation, self.hidden_dim * self.rdimMul))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        if 'newModEl' in models_list:
            # Add additional entity or relation embeddings required by your model.
            pass


        self.load_choice_embs = load_choice_embs #False  # If True, SMART load pre-trained relation weights
        self.wnrr_embds = "wn18rr_embds.npy" # "wn18rr_egt_embds.npy" #
        self.yt_embds = "youtube_embds.npy"
        if self.load_choice_embs:
            self.choice_embedding = nn.Parameter(torch.Tensor(np.load(self.wnrr_embds)), requires_grad=False)
        else:
            self.choice_embedding = nn.Parameter(torch.ones(num_relation, len(self.models)))

        self.single_egt = single_egt
        self.single_egt_vals = torch.Tensor([0]).cuda()
        self.pi = 3.14159262358979323846

        self.freeze = False
        self.start = False
        self.initiate = True
        self.step = 0
        self.threshold = threshold
        
    def func(self, head, rel, tail, batch_type, relation_choice, sample):
        '''
        This model uses EGTs (e.g. translation, rotation, reflection, and scaling)  provided by the user as basis of relation embeddings.
        Each relation r_i learns its optimal embedding through the 4 values of relation_choice[i] which are softmaxed.
        '''

        pi = 3.14159265358979323846

        if self.initiate:
            select = torch.ones_like(relation_choice)
        if self.start:
            self.initiate = False
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
        if self.freeze:
            self.initiate = False
            self.start = False
            if self.load_choice_embs:
                select = relation_choice
            else:
                select = torch.sqrt(torch.softmax(relation_choice, dim=-1))
            save, _ = select.max(dim=2, keepdim=True)
            if self.threshold > 0:
                select = torch.where(select < torch.tensor([self.threshold]).cuda(), torch.tensor([0]).cuda(), save)
            else:
                select = torch.where(select < save, torch.tensor([0]).cuda(), torch.tensor([1]).cuda())

            if self.single_egt:
                if batch_type == BatchType.SINGLE:
                    index = sample[:, 1]
                elif batch_type == BatchType.HEAD_BATCH:
                    head_part, tail_part = sample
                    index = head_part[:, 1]
                else:
                    head_part, tail_part = sample
                    index = head_part[:, 1]
                single_egt_vals = self.choice_embedding.clone()
                single_egt = single_egt_vals.sum(dim=0, keepdim=True)
                single_egt_ind = single_egt.argsort()[0][-1].item()
                single_egt_val = single_egt[0][single_egt_ind]
                single_egt_vals[:, single_egt_ind] = single_egt_vals[:, single_egt_ind] + single_egt_val
                self.single_egt_vals = single_egt_vals
                select = torch.index_select(single_egt_vals, dim=0, index=index).unsqueeze(1)

        ## Transformations
        score = 0
        for idx, model in enumerate(self.models):
            if idx == 0:
                rel_ = rel[:, :, :self.saveRdimMul[idx]*self.hidden_dim]
            else:
                rel_ = rel[:, :, self.saveRdimMul[idx-1]*self.hidden_dim : self.saveRdimMul[idx]*self.hidden_dim]
            sel = select[:, :, idx].view((select.shape[0], -1, 1))
            score_, re_rhead, im_rhead = model(head, rel_, tail, batch_type, sel)
            score = score + score_
        return self.gamma.item() + score



