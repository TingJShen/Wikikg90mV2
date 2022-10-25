# -*- coding: utf-8 -*-
#
# score_fun.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT
import numpy as np


def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = th.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res


def batched_l1_dist(a, b):
    res = th.cdist(a, b, p=1)
    return res


class TransEScore(nn.Module):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """

    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else:  # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        #import pdb; pdb.set_trace()
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)

        return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        #import pdb; pdb.set_trace()
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn


class TransRScore(nn.Module):
    """TransR score function
    Paper link: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523
    """

    def __init__(self, gamma, projection_emb, relation_dim, entity_dim):
        super(TransRScore, self).__init__()
        self.gamma = gamma
        self.projection_emb = projection_emb
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.data['head_emb']
        tail = edges.data['tail_emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        g.edata['head_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][head_ids], projection)
        g.edata['tail_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][tail_ids], projection)

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, -1, 1, self.entity_dim)
                tail = th.matmul(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                head = head.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                head = th.matmul(head, projection)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, -1, 1, self.entity_dim)
                head = th.matmul(head, projection)
                head = head.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                tail = tail.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                tail = th.matmul(tail, projection)
                return head, tail
            return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        self.projection_emb.init(1.0)

    def update(self, gpu_id=-1):
        self.projection_emb.update(gpu_id)

    def save(self, path, name):
        self.projection_emb.save(path, name + 'projection')

    def load(self, path, name):
        self.projection_emb.load(path, name + 'projection')

    def prepare_local_emb(self, projection_emb):
        self.global_projection_emb = self.projection_emb
        self.projection_emb = projection_emb

    def prepare_cross_rels(self, cross_rels):
        self.projection_emb.setup_cross_rels(cross_rels, self.global_projection_emb)

    def writeback_local_emb(self, idx):
        self.global_projection_emb.emb[idx] = self.projection_emb.emb.cpu()[idx]

    def load_local_emb(self, projection_emb):
        device = projection_emb.emb.device
        projection_emb.emb = self.projection_emb.emb.to(device)
        self.projection_emb = projection_emb

    def share_memory(self):
        self.projection_emb.share_memory()

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                heads = heads - relations
                heads = heads.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn


class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """

    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb * rel_emb).unsqueeze(2) * tail_emb.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)

            return fn


class ComplExScore(nn.Module):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """

    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
            + img_head * img_tail * real_rel \
            + real_head * img_tail * img_rel \
            - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        real_head, img_head = th.chunk(head_emb, 2, dim=-1)
        real_tail, img_tail = th.chunk(tail_emb, 2, dim=-1)
        real_rel, img_rel = th.chunk(rel_emb, 2, dim=-1)

        score = (real_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0) \
            + (img_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
            + (real_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
            - (img_head.unsqueeze(1) * img_rel.unsqueeze(0)
               ).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                return th.bmm(tmp, tails)
            return fn


class RESCALScore(nn.Module):
    """RESCAL score function
    Paper link: http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """

    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1).unsqueeze(1)
        rel_emb = rel_emb.view(-1, self.relation_dim, self.entity_dim)
        score = head_emb * th.einsum('abc,dc->adb', rel_emb, tail_emb).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tails = tails.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, tails).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                heads = heads.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, heads).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)
            return fn


class RotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """

    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):

        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        re_head, im_head = th.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = th.chunk(tail_emb, 2, dim=-1)

        phase_rel = rel_emb / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head.unsqueeze(1) * re_rel.unsqueeze(0) - \
            im_head.unsqueeze(1) * im_rel.unsqueeze(0)
        im_score = re_head.unsqueeze(1) * im_rel.unsqueeze(0) + \
            im_head.unsqueeze(1) * re_rel.unsqueeze(0)

        re_score = re_score.unsqueeze(2) - re_tail.unsqueeze(0).unsqueeze(0)
        im_score = im_score.unsqueeze(2) - im_tail.unsqueeze(0).unsqueeze(0)
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - heads
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real

                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)

            return fn


class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """

    def __init__(self):
        super(SimplEScore, self).__init__()

    def edge_func(self, edges):
        head_i, head_j = th.chunk(edges.src['emb'], 2, dim=-1)
        tail_i, tail_j = th.chunk(edges.dst['emb'], 2, dim=-1)
        rel, rel_inv = th.chunk(edges.data['emb'], 2, dim=-1)
        forward_score = head_i * rel * tail_j
        backward_score = tail_i * rel_inv * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = th.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_i, head_j = th.chunk(head_emb.unsqueeze(1), 2, dim=-1)
        tail_i, tail_j = th.chunk(tail_emb.unsqueeze(0).unsqueeze(0), 2, dim=-1)
        rel, rel_inv = th.chunk(rel_emb.unsqueeze(0), 2, dim=-1)
        forward_tmp = (head_i * rel).unsqueeze(2) * tail_j
        backward_tmp = (head_j * rel_inv).unsqueeze(2) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return th.sum(score, dim=-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tail_i = tails[..., :hidden_dim // 2]
                tail_j = tails[..., hidden_dim // 2:]
                rel = relations[..., : hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (rel * tail_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * tail_i).reshape(num_chunks, chunk_size, hidden_dim//2)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                head_i = heads[..., :hidden_dim // 2, :]
                head_j = heads[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, head_i) + th.bmm(backward_tmp, head_j))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                head_i = heads[..., :hidden_dim // 2]
                head_j = heads[..., hidden_dim // 2:]
                rel = relations[..., :hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (head_i * rel).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * head_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tail_i = tails[..., :hidden_dim // 2, :]
                tail_j = tails[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, tail_j) + th.bmm(backward_tmp, tail_i))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn


class PairRotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """

    def __init__(self, gamma, emb_init):
        super(PairRotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_relation_head, re_relation_tail = th.chunk(re_rel, 2, dim=-1)
        im_relation_head, im_relation_tail = th.chunk(im_rel, 2, dim=-1)
        head_re_score = re_head * re_relation_head - im_head * im_relation_head
        head_im_score = re_head * im_relation_head + im_head * re_relation_head
        tail_re_score = re_tail * re_relation_tail - im_tail * im_relation_tail
        tail_im_score = re_tail * im_relation_tail + im_tail * re_relation_tail
        re_score = head_re_score - tail_re_score
        im_score = head_im_score - tail_im_score
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        raise NotImplementedError('PairRotatE infer not exists!')

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                head_rel_real, tail_rel_real = rel_real[...,
                                                        :hidden_dim // 2], rel_real[..., hidden_dim // 2:]
                head_rel_imag, tail_rel_imag = rel_imag[...,
                                                        :hidden_dim // 2], rel_imag[..., hidden_dim // 2:]

                head_emb_real = heads[..., :hidden_dim // 2]
                head_emb_imag = heads[..., hidden_dim // 2:]
                head_real = head_emb_real * head_rel_real - head_emb_imag * head_rel_imag
                head_imag = head_emb_real * head_rel_imag + head_emb_imag * head_rel_real
                head_emb_complex = th.cat((head_real, head_imag), dim=-1)

                tail_emb_real = tails[..., :hidden_dim // 2]
                tail_emb_imag = tails[..., hidden_dim // 2:]
                tail_real = tail_emb_real * tail_rel_real - tail_emb_imag * tail_rel_imag
                tail_imag = tail_emb_real * tail_rel_imag + tail_emb_imag * tail_rel_real
                tail_emb_complex = th.cat((tail_real, tail_imag), dim=-1)

                tails_tmp = tail_emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads_tmp = head_emb_complex.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = heads_tmp - tails_tmp
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                ar1, ar2 = rel_real[..., :hidden_dim // 2], rel_real[..., hidden_dim // 2:]
                br1, br2 = rel_imag[..., :hidden_dim // 2], rel_imag[..., hidden_dim // 2:]
                ah, bh = heads[..., :hidden_dim // 2], heads[..., hidden_dim // 2:]

                head_real = ah * ar1 * ar2 - bh * br1 * ar2 + ah * br1 * br2 + bh * ar1 * br2
                head_imag = -ah * ar1 * br2 + bh * br1 * br2 + ah * br1 * ar2 + bh * ar1 * ar2
                head_emb_complex = th.cat((head_real, head_imag), dim=-1)
                tmp = head_emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)

                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)
            return fn


class PairREScore(nn.Module):
    def __init__(self, gamma):
        self.gamma = gamma
        super(PairREScore, self).__init__()

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def prepare(self, g, gpu_id, trace=False):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def edge_func(self, edges):
        head = edges.src['emb']
        relation = edges.data['emb']
        tail = edges.dst['emb']

        re_head, re_tail = th.chunk(relation, 2, dim=-1)

        head = functional.normalize(head, 2, -1)
        tail = functional.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma - th.norm(score, p=1, dim=-1)

        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        #head_emb = head_emb.unsqueeze(1)
        #rel_emb = rel_emb.unsqueeze(0)
        #score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)
        #return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)
        raise NotImplementedError('PairRE infer not exists!')

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                re_head = relations[..., :hidden_dim]
                re_tail = relations[..., hidden_dim:]

                heads = functional.normalize(heads, 2, -1)
                tails = functional.normalize(tails, 2, -1)

                tails_part = tails * re_tail
                tails_part = tails_part.reshape(num_chunks, chunk_size, 1, hidden_dim)

                re_head = re_head.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                heads_part = heads * re_head

                score = self.gamma - th.norm(tails_part-heads_part, p=1, dim=-1)

                return score

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                re_head = relations[..., :hidden_dim]
                re_tail = relations[..., hidden_dim:]

                heads = functional.normalize(heads, 2, -1)
                tails = functional.normalize(tails, 2, -1)

                heads_part = heads * re_head
                heads_part = heads_part.reshape(num_chunks, chunk_size, 1, hidden_dim)
                re_tail = re_tail.reshape(num_chunks, chunk_size, 1, hidden_dim)

                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                tails_part = tails * re_tail

                score = self.gamma - th.norm(heads_part-tails_part, p=1, dim=-1)

                return score

            return fn

class OTEScore(nn.Module):
    def __init__(self, gamma, num_elem, scale_type=0):
        super(OTEScore, self).__init__()
        self.gamma = gamma
        self.num_elem = num_elem
        self.scale_type = scale_type

    def forward_rel_index(self, inputs, inputs_rel, rel_idx, eps=1e-6):
        # inputs: * X num_dim, where num_dim % num_elem == 0
        inputs_size = inputs.size()
        # assert inputs_size[:-1] == inputs_rel.size()[:-1]
        num_dim = inputs.size(-1)

        inputs = inputs.view(-1, 1, self.num_elem)
        if self.use_scale:
            rel_size = inputs_rel.size()
            rel = inputs_rel.view(-1, self.num_elem, self.num_elem + 1)
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(dim=-1, keepdim=True)
            rel = rel[:, :, :self.num_elem] * scale
            rel = rel.view(rel_size[0], -1, self.num_elem,
                           self.num_elem).index_select(0, rel_idx)
            rel = rel.view(-1, self.num_elem, self.num_elem)
            outputs = torch.bmm(inputs, rel)
        else:
            rel = inputs_rel.index_select(0, rel_idx)
            rel = rel.view(-1, self.num_elem, self.num_elem)
            outputs = torch.bmm(inputs, rel)
        return outputs.view(inputs_size)

    @property
    def use_scale(self):
        return self.scale_type > 0

    def score(self, inputs, inputs_rel, inputs_last):
        inputs_size = inputs.size()
        assert inputs_size[:-1] == inputs_rel.size()[:-1]
        num_dim = inputs.size(-1)
        inputs = inputs.view(-1, 1, self.num_elem)
        if self.use_scale:
            rel = inputs_rel.view(-1, self.num_elem, self.num_elem + 1)
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(dim=-1, keepdim=True)
            rel_scale = rel[:, :, :self.num_elem] * scale
            outputs = torch.bmm(inputs, rel_scale)
        else:
            rel = inputs_rel.view(-1, self.num_elem, self.num_elem)
            outputs = torch.bmm(inputs, rel)
        outputs = outputs.view(inputs_size)
        outputs = outputs - inputs_last
        outputs_size = outputs.size()
        num_dim = outputs.size(-1)
        outputs = outputs.view(-1, self.num_elem)
        scores = outputs.norm(dim=-1).view(-1, num_dim // self.num_elem).sum(
            dim=-1).view(outputs_size[:-1])
        return scores

    def neg_score(self, inputs, inputs_rel, inputs_last, neg_sample_size,
                  chunk_size):
        inputs_size = inputs.size()
        assert inputs_size[:-1] == inputs_rel.size()[:-1]
        num_dim = inputs.size(-1)
        inputs = inputs.view(-1, 1, self.num_elem)
        if self.use_scale:
            rel = inputs_rel.view(-1, self.num_elem, self.num_elem + 1)
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(dim=-1, keepdim=True)
            rel_scale = rel[:, :, :self.num_elem] * scale
            outputs = torch.bmm(inputs, rel_scale)
        else:
            rel = inputs_rel.view(-1, self.num_elem, self.num_elem)
            outputs = torch.bmm(inputs, rel)
        outputs = outputs.view(-1, chunk_size, 1, inputs_size[-1])
        inputs_last = inputs_last.view(-1, 1, neg_sample_size, inputs_size[-1])
        outputs = outputs - inputs_last
        outputs_size = outputs.size()
        num_dim = outputs.size(-1)
        outputs = outputs.view(-1, num_dim).view(-1, num_dim // self.num_elem,
                                                 self.num_elem)
        outputs = outputs.view(-1, self.num_elem)
        scores = outputs.norm(dim=-1).view(-1, num_dim // self.num_elem).sum(
            dim=-1).view(outputs_size[:-1])
        return scores

    def get_scale(self, scale):
        if self.scale_type == 1:
            return scale.abs()
        if self.scale_type == 2:
            return scale.exp()
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def reverse_scale(self, scale, eps=1e-9):
        if self.scale_type == 1:
            return 1 / (abs(scale) + eps)
        if self.scale_type == 2:
            return -scale
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def scale_init(self):
        if self.scale_type == 1:
            return 1.0
        if self.scale_type == 2:
            return 0.0
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def orth_embedding(self, embeddings, eps=1e-18, do_test=True):
        # orthogonormalizing embeddings
        # embeddings: num_emb X num_elem X (num_elem + (1 or 0))
        num_emb = embeddings.size(0)
        assert embeddings.size(1) == self.num_elem
        assert embeddings.size(2) == (self.num_elem +
                                      (1 if self.use_scale else 0))
        if self.use_scale:
            emb_scale = embeddings[:, :, -1]
            embeddings = embeddings[:, :, :self.num_elem]

        u = [embeddings[:, 0]]
        uu = [0] * self.num_elem
        uu[0] = (u[0] * u[0]).sum(dim=-1)
        if do_test and (uu[0] < eps).sum() > 1:
            return None
        u_d = embeddings[:, 1:]
        for i in range(1, self.num_elem):
            u_d = u_d - u[-1].unsqueeze(dim=1) * (
                (embeddings[:, i:] * u[i - 1].unsqueeze(dim=1)).sum(
                    dim=-1) / uu[i - 1].unsqueeze(dim=1)).unsqueeze(-1)
            u_i = u_d[:, 0]
            u_d = u_d[:, 1:]
            uu[i] = (u_i * u_i).sum(dim=-1)
            if do_test and (uu[i] < eps).sum() > 1:
                return None
            u.append(u_i)

        u = torch.stack(u, dim=1)  # num_emb X num_elem X num_elem
        u_norm = u.norm(dim=-1, keepdim=True)
        u = u / u_norm
        if self.use_scale:
            u = torch.cat((u, emb_scale.unsqueeze(-1)), dim=-1)
        return u

    def orth_rel_embedding(self, relation_embedding):
        rel_emb_size = relation_embedding.size()
        ote_size = self.num_elem
        scale_dim = 1 if self.use_scale else 0
        rel_embedding = relation_embedding.view(-1, ote_size,
                                                ote_size + scale_dim)
        rel_embedding = self.orth_embedding(rel_embedding).view(rel_emb_size)
        return rel_embedding

    def orth_reverse_mat(self, rel_embeddings):
        rel_size = rel_embeddings.size()
        if self.use_scale:
            rel_emb = rel_embeddings.view(-1, self.num_elem, self.num_elem + 1)
            rel_mat = rel_emb[:, :, :self.num_elem].contiguous().transpose(1,
                                                                           2)
            rel_scale = self.reverse_scale(rel_emb[:, :, self.num_elem:])
            rel_embeddings = torch.cat((rel_mat, rel_scale),
                                       dim=-1).view(rel_size)
        else:
            rel_embeddings = rel_embeddings.view(
                -1, self.num_elem, self.num_elem).transpose(
                    1, 2).contiguous().view(rel_size)
        return rel_embeddings

    def edge_func(self, edges, neg_head):
        heads = edges.src['emb']
        tails = edges.dst['emb']
        relations = edges.data['emb']
        # get the orth relation embedding
        relations = self.orth_rel_embedding(relations)
        if neg_head:
            relations = self.orth_reverse_mat(relations)
            score_result = self.score(tails, relations, heads)
        else:
            score_result = self.score(heads, relations, tails)
        score_result = self.gamma - score_result
        return {'score': score_result}

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        # print for debug
        # print(type(g))
        # print(g.__dict__)
        g.apply_edges(lambda edges: self.edge_func(edges, g.neg_head))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:

            def fn(heads, relations, tails, num_chunks, chunk_size,
                   neg_sample_size):
                relations = self.orth_rel_embedding(relations)
                relations = self.orth_reverse_mat(relations)
                score_result = self.neg_score(tails, relations, heads,
                                              neg_sample_size, chunk_size)
                score_result = gamma - score_result
                return score_result

            return fn
        else:

            def fn(heads, relations, tails, num_chunks, chunk_size,
                   neg_sample_size):
                relations = self.orth_rel_embedding(relations)
                score_result = self.neg_score(heads, relations, tails,
                                              neg_sample_size, chunk_size)
                score_result = gamma - score_result
                return score_result

            return fn

class AutoScore(nn.Module):
    """Auto score function
       https://arxiv.org/abs/1904.11682
    """

    def __init__(self):
        super(AutoScore, self).__init__()
        self.struct = [
            3, 1, 2, 0, 2, 0, 1, -1, 2, 1, 0, 1, 2, 3, 2, -1, 2, 2, 3, 1
        ]

    def get_hr(self, heads, relations):
        idx = tuple(self.struct)
        h1, h2, h3, h4 = torch.chunk(heads, 4, dim=-1)
        r1, r2, r3, r4 = torch.chunk(relations, 4, dim=-1)

        hs = [h1, h2, h3, h4]
        rs = [r1, r2, r3, r4]

        vs = [0, 0, 0, 0]
        vs[idx[0]] = h1 * r1
        vs[idx[1]] = h2 * r2
        vs[idx[2]] = h3 * r3
        vs[idx[3]] = h4 * r4

        res_B = (len(idx) - 4) // 4
        for b_ in range(1, res_B + 1):
            base = 4 * b_
            vs[idx[base + 2]] += rs[idx[base + 0]] * hs[idx[base + 1]] * int(
                idx[base + 3])
        return torch.cat(vs, 1)

    def get_rt(self, tails, relations):
        idx = tuple(self.struct)
        t1, t2, t3, t4 = torch.chunk(tails, 4, dim=-1)
        r1, r2, r3, r4 = torch.chunk(relations, 4, dim=-1)

        ts = [t1, t2, t3, t4]
        rs = [r1, r2, r3, r4]

        vs = [
            r1 * ts[idx[0]], r2 * ts[idx[1]], r3 * ts[idx[2]], r4 * ts[idx[3]]
        ]

        res_B = (len(idx) - 4) // 4
        for b_ in range(1, res_B + 1):
            base = 4 * b_
            vs[idx[base + 1]] += rs[idx[base + 0]] * ts[idx[base + 2]] * int(
                idx[base + 3])
        return torch.cat(vs, 1)

    def edge_func(self, edges):
        heads = edges.src['emb']
        tails = edges.dst['emb']
        relations = edges.data['emb']
        vec_hr = self.get_hr(heads, relations)
        score = torch.sum(vec_hr * tails, -1)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def test_tail(self, head, rela):
        vec_hr = self.get_hr(head, rela)
        tail_embed = self.ent_embed.weight
        scores = torch.mm(vec_hr, tail_embed.transpose(1, 0))
        return scores

    def test_head(self, rela, tail):
        vec_rt = self.get_rt(rela, tail)
        head_embed = self.ent_embed.weight
        scores = torch.mm(vec_rt, head_embed.transpose(1, 0))
        return scores

    def create_neg(self, neg_head):
        if neg_head:

            def fn(heads, relations, tails, num_chunks, chunk_size,
                   neg_sample_size):
                hidden_dim = tails.shape[1]
                vec_rt = self.get_rt(tails, relations)
                vec_rt = vec_rt.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = heads.transpose(1, 2)
                score = th.bmm(vec_rt, heads)
                return score

            return fn
        else:

            def fn(heads, relations, tails, num_chunks, chunk_size,
                   neg_sample_size):
                hidden_dim = heads.shape[1]
                vec_hr = self.get_hr(heads, relations)
                vec_hr = vec_hr.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails.transpose(1, 2)
                score = th.bmm(vec_hr, tails)
                return score

            return fn