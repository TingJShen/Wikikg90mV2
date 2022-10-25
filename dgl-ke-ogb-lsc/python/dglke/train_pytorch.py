# -*- coding: utf-8 -*-
#
# train_pytorch.py
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

from math import ceil
import pickle
from tqdm import tqdm
from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
from collections import defaultdict
import pdb
from .dataloader import get_dataset
from .dataloader import EvalDataset
import dgl.backend as F
from dgl.contrib import KVClient
import dgl
from functools import wraps
import time
import logging
import os
from .utils import save_model, get_compatible_batch_size
from .models import KEModel
from .models.pytorch.tensor_models import thread_wrapped_func
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import torch

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")


def load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel,
                    ent_feat_dim=ent_feat_dim, rel_feat_dim=rel_feat_dim,
                    LRE=args.LRE, LRE_rank=args.LRE_rank,
                    feat_hidden_dim=args.feat_hidden_dim)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

def load_model_from_checkpoint(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim):
    model = load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim)
    model.load_emb(args.load_path,args.dataset,args.index)
    return model
#def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path, ent_feat_dim, rel_feat_dim):
#    model = load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim)
#    model.load_emb(ckpt_path, args.dataset)
    # 这边百度还load了mlp参数
    # print("load embedding done.")
    # model.transform_net.load_parameters(
    #     os.path.join(ckpt_path, args.dataset + "_" + args.model_name + "_mlp"))
    # print("load mlp parameters done.")
#    return model


def train(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank %
                          len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    if args.encoder_model_name in ['roberta', 'concat', 'concat_v1']:
        model.transform_net = model.transform_net.to(th.device('cuda:' + str(gpu_id)))
        optimizer = th.optim.Adam(model.transform_net.parameters(), args.mlp_lr)
    else:
        optimizer = None

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    # 百度加的两个变量
    # best_valid_mrr = -1
    # best_valid_idx = -1
    for step in tqdm(range(0, args.max_step)):
        # print("start step", step)
        start1 = time.time()
        pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        if optimizer is not None:
            optimizer.zero_grad()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        if rank == 0 :
            fp = open(args.save_path+'loss.txt', 'a')
            fp.write(str(loss.item())+'\n')
            #print(sum(p.numel() for p in model.transform_net.parameters()))
            #print(str(loss.item()))
            fp.close()

        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)
        if optimizer is not None:
            optimizer.step()

        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
                (step + 1) % args.force_sync_interval == 0:
            # 百度加了barrier is not None的条件
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    logging.info('[proc {}][Train]({}/{}) average {}: {}'.format(rank,
                                                                                 (step + 1), args.max_step, k, v))
                logs = []
                logging.info('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                                    time.time() - start))
                logging.info('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()
        # 百度在这里设置了学习率衰减
        # if args.lr_decay_rate is not None and (step + 1
        #                                        ) % args.lr_decay_interval == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * args.lr_decay_rate
        # if True:
        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            logging.info('[proc {}]barrier wait in validation take {:.3f} seconds:'.format(
                rank, time.time() - valid_start))
            valid_start = time.time()
            if valid_samplers is not None:
                # if step + 1 == args.max_step:
                #    valid_input_dict = test_full(args, model, valid_samplers, step, rank, mode='Valid')
                # else:
                valid_input_dict = test(args, model, valid_samplers, step, rank, mode='Valid')
                th.save(valid_input_dict, os.path.join(
                    args.save_path, "valid_{}_{}.pkl".format(rank, step)))
            if test_samplers is not None:
                test_input_dict = test(args, model, test_samplers, step, rank, mode='Test')
                th.save(test_input_dict, os.path.join(
                    args.save_path, "test_{}_{}.pkl".format(rank, step)))
            logging.info('[proc {}]validation and test take {:.3f} seconds:'.format(
                rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()

        if rank == 0  and (step + 1) % args.save_step == 0 and step > 1 :
            time.sleep(10)
            save_model(args, model, None, None,step)
            print('proc {} model saved'.format(rank))
            # 百度加了这一段
            # if rank == 0:
            #     valid_result_dict = defaultdict(lambda: defaultdict(list))
            #     for proc in range(args.num_proc):
            #         valid_result_dict_proc = th.load(
            #             os.path.join(args.save_path,
            #                          "valid_{}_{}.pkl".format(proc, step)),
            #             map_location=th.device('cpu'))
            #         for result_dict_proc, result_dict in zip(
            #             [valid_result_dict_proc], [valid_result_dict]):
            #             for key in result_dict_proc['h,r->t']:
            #                 result_dict['h,r->t'][key].append(result_dict_proc[
            #                     'h,r->t'][key].numpy())
            #     for result_dict in [valid_result_dict]:
            #         for key in result_dict['h,r->t']:
            #             result_dict['h,r->t'][key] = np.concatenate(
            #                 result_dict['h,r->t'][key], 0)
            #
            #     evaluator = WikiKG90MEvaluator()
            #     metrics = evaluator.eval(valid_result_dict)
            #     metric = 'mrr'
            #     logging.info("valid-{} at step {}: {}".format(metric, step,
            #                                                   metrics[metric]))
            #     if metrics[metric] > best_valid_mrr:
            #         best_valid_mrr = metrics[metric]
            #         best_valid_idx = step
            #         if best_valid_mrr > args.save_threshold:
            #             save_model(args, model, None, None)
            #             logging.info('proc {} model saved'.format(rank))
            # if barrier is not None:
            #     barrier.wait()
    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    time.sleep(10)
    if rank == 0 and not args.no_save_emb:
        save_model(args, model, None, None,str(args.max_step))
        print('proc {} model saved'.format(rank))

    if barrier is not None:
        barrier.wait()
    print('proc {} after barrier'.format(rank))
    if args.async_update:
        model.finish_async_update()
    print('proc {} finish async update'.format(rank))
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)
    print('proc {} return'.format(rank))


def test_full(args, model, test_samplers, step, rank=0, mode='Test'):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank %
                          len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))
    # PIE加的
    all_candidate = torch.arange(model.n_entities)
    chunk_size = 100000
    chunks = int(model.n_entities / chunk_size)
    # PIE加的
    with th.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        for sampler in test_samplers:
            for query, ans, candidate in tqdm(sampler, disable=not args.print_on_screen, total=ceil(sampler.num_edges/sampler.batch_size)):
                for i in range(query.shape[0]):
                    scores = []
                    q = query[i].unsqueeze(0)
                    a = ans[i].unsqueeze(0)

                    for c in range(chunks):
                        log = model.forward_test_wikikg(q, a,
                                                        all_candidate[c*chunk_size: (c+1)*chunk_size].unsqueeze(0), sampler.mode, gpu_id).cpu()
                    scores.append(log)
                    log = model.forward_test_wikikg(
                        q, a, all_candidate[c*chunk_size:].unsqueeze(0), sampler.mode, gpu_id).cpu()
                    scores.append(log)
                    _, topk = torch.topk(torch.cat(scores, dim=1), 50, dim=1)
                    logs[sampler.mode].append(topk)
                    answers[sampler.mode].append(a)
        print("[{}] finished {} forward".format(rank, mode))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            input_dict['h,r->t'] = {'t': th.cat(
                answers['h,r->t'], 0), 't_pred_top10': th.cat(logs['h,r->t'], 0)}
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()

    return input_dict


def test(args, model, test_samplers, step, rank=0, mode='Test'):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank %
                          len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = defaultdict(list)
        scores = defaultdict(list)
        answers = defaultdict(list)
        for sampler in test_samplers:
            for query, ans, candidate in tqdm(sampler, disable=not args.print_on_screen, total=ceil(sampler.num_edges/sampler.batch_size)):
                log = model.forward_test_wikikg(query, ans, candidate, sampler.mode, gpu_id).cpu()
                scores[sampler.mode].append(log)
                logs[sampler.mode].append(F.argsort(log, dim=1, descending=True)[:, :10])
                answers[sampler.mode].append(ans)
        print("[{}] finished {} forward".format(rank, mode))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            input_dict['h,r->t'] = {'t': th.cat(
                answers['h,r->t'], 0), 't_pred_top10': th.cat(logs['h,r->t'], 0),
                't_pred_score': th.cat(scores['h,r->t'], 0)}
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()

    return input_dict


@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers,
          test_samplers, rank, rel_parts, cross_rels, barrier)


@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, 0, rank, mode)
