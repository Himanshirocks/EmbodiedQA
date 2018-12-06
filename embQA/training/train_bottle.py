import time
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from models_bottle import NavCnnModel, NavCnnRnnModel, NavCnnRnnMultModel, NavPlannerControllerModel, BottleSlaveRnn, BottleMasterGRU
from data import EqaDataLoader
from metrics import NavMetric
from models import MaskedNLLCriterion
from models import get_state, ensure_shared_grads
from data import load_vocab
from torch.autograd import Variable
from tqdm import tqdm
import time
import cv2
torch.backends.cudnn.enabled = False
import pdb
import pickle as pkl
from collections import defaultdict
import csv

def oneHot(vec, dim):
    batch_size = vec.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), vec.long()] = 1
    return out

def load_semantic_classes(color_file):
    if color_file is None:
        raise ValueError('please input colormap_fine.csv file')

    semantic_classes = {}

    with open(color_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # OpenCV is in BGR Format
                c = np.array((row['r'], row['g'], row['b']), dtype=np.uint8)
                fine_cat = row['name'].lower()
                semantic_classes[fine_cat] = c

    return semantic_classes

def coverage(img, target_obj_class, semantic_classes):
    
    hei, wid, _ = img.shape
    trgt_obj_col = semantic_classes[target_obj_class]
    mask = np.all(img == trgt_obj_col, axis=2)
    cov = np.sum(mask)/(hei * wid)
    
    return cov

def avgCov(cov_dict):
    """ calculate avg coverage for the last 5 frames for all batches"""
    total = 0
    count = 0
    
    for t in list(cov_dict.keys()):
        for num in list(cov_dict[t].keys()):
            for i in list(cov_dict[t][num].keys()):
                cov_epi = cov_dict[t][num][i][-6:-1]
                total += np.sum(np.array(cov_epi))
                count += 1
    
    return total/count

def oneHot_vol(vec, dim):
    batch_size, T_p  = vec.size(0)
    out = torch.zeros(batch_size, T_p, dim)
    for i in range(batch_size):
        out[i, np.arange(T_p), vec[i].long()] = 1
    # out[np.arange(batch_size), vec.long()] = 1
    return out

################################################################################################
#make models trained in pytorch 4 compatible with earlier pytorch versions
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

################################################################################################

def eval(rank, args, best_eval_acc, best_cov, model=None, epoch=0):

    #torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_kwargs = {
        'question_input': True,
        'question_vocab': load_vocab(args.vocab_json)
    }
    model = NavCnnModel(**model_kwargs)

    eval_loader_kwargs = {
        'questions_h5': getattr(args, args.eval_split + '_h5'),
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
        'map_resolution': args.map_resolution,
        'batch_size': 1,
        'input_type': args.model_type,
        'num_frames': 5,
        'split': args.eval_split,
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': 0,
        'to_cache': False,
        'overfit': args.overfit,
        'max_controller_actions': args.max_controller_actions,
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))
    logging.info("EVAL: eval_loader has {} samples".format(len(eval_loader.dataset)))

    #Saty:
    semantic_classes = load_semantic_classes(eval_loader.dataset.cfg['colorFile'])
    coverage_log = defaultdict(dict)
    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')
    invalids = []

    model.load_state_dict(shared_model.state_dict())
    model.eval()

    # that's a lot of numbers
    metrics = NavMetric(
        info={'split': args.eval_split,
              'thread': rank},
        metric_names=[
            'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
            'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
            'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
            'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
            'ep_len_30', 'ep_len_50'
        ],
        log_json=args.output_log_path)

    if 'cnn' in args.model_type:

        done = False
        t =0 
        while done == False:

            for num, batch in enumerate(tqdm(eval_loader)):

                model.load_state_dict(shared_model.state_dict())
                model.to(device)

                idx, questions, _, img_feats, actions_in, actions_out, actions_out_prev, pos_queue_curr, action_length = batch
                
                metrics_slug = {}

                # evaluate at multiple initializations
                for i in [10, 30, 50]:

                    if action_length[0] + 1 - i - 5 < 0:
                        invalids.append(idx[0])
                        continue

                    ep_inds = [x for x in range(action_length[0] + 1 - i - 5, action_length[0] + 1 - i)]

                    sub_img_feats = torch.index_select(img_feats, 1, torch.LongTensor(ep_inds))

                    init_pos = eval_loader.dataset.episode_pos_queue[ep_inds[-1]]

                    h3d = eval_loader.dataset.episode_house

                    h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                    init_dist_to_target = h3d.get_dist_to_target(h3d.env.cam.pos)
                    if init_dist_to_target < 0:  # unreachable
                        invalids.append(idx[0])
                        continue

                    sub_img_feats_var = Variable(sub_img_feats.to(device))
                    if '+q' in args.model_type:
                        questions_var = Variable(questions.to(device))
                    actions_out = actions_out.cuda()
                    actions_out_prev = actions_out_prev.cuda()
                    pos_queue_curr = Variable(pos_queue_curr.cuda())
                    actions_out_prev_OH = Variable(oneHot(actions_out_prev, 3).cuda())


                    # sample actions till max steps or <stop>
                    # max no. of actions = 100
                    episode_length = 0
                    episode_done = True

                    dists_to_target, pos_queue, actions = [init_dist_to_target], [init_pos], []
                    cov_batch_i = []

                    for step in range(args.max_episode_length):

                        episode_length += 1

                        if '+q' in args.model_type:
                            scores = model(sub_img_feats_var,
                                           questions_var)
                        else:
                            scores = model(sub_img_feats_var)

                        prob = F.softmax(scores, dim=1)

                        action = int(prob.max(1)[1].data.cpu().numpy()[0])

                        actions.append(action)

                        img, _, episode_done = h3d.step(action)

                        episode_done = episode_done or episode_length >= args.max_episode_length

                        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                        
                        img_feat_var = eval_loader.dataset.cnn(
                            Variable(img.view(1, 3, 224, 224).to(device))).view(1, 1, 3200)
                        
                        sub_img_feats_var = torch.cat([sub_img_feats_var, img_feat_var], dim=1)
                        sub_img_feats_var = sub_img_feats_var[:, -5:, :]

                        dists_to_target.append(h3d.get_dist_to_target(h3d.env.cam.pos))
                        
                        pos_queue.append([
                            h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                            h3d.env.cam.pos.z, h3d.env.cam.yaw
                        ])
                        target_obj = eval_loader.dataset.target_obj['fine_class']
                        img_semantic = h3d.env.render(mode='semantic')
                        cov = coverage(img_semantic, target_obj, semantic_classes)
                        cov_batch_i.append(cov)
                        

                        if episode_done:
                            cov_batch_i.append(target_obj)
                            if num not in coverage_log[t].keys():
                                coverage_log[t][num] = {}
                            coverage_log[t][num].update({i:cov_batch_i})    
                            break

                    # compute stats
                    metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                    metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                    metrics_slug['d_D_' + str(
                        i)] = dists_to_target[0] - dists_to_target[-1]
                    metrics_slug['d_min_' + str(i)] = np.array(
                        dists_to_target).min()
                    metrics_slug['ep_len_' + str(i)] = episode_length
                    if action == 3:
                        metrics_slug['stop_' + str(i)] = 1
                    else:
                        metrics_slug['stop_' + str(i)] = 0
                    inside_room = []
                    for p in pos_queue:
                        inside_room.append(
                            h3d.is_inside_room(
                                p, eval_loader.dataset.target_room))
                    if inside_room[-1] == True:
                        metrics_slug['r_T_' + str(i)] = 1
                    else:
                        metrics_slug['r_T_' + str(i)] = 0
                    if any([x == True for x in inside_room]) == True:
                        metrics_slug['r_e_' + str(i)] = 1
                    else:
                        metrics_slug['r_e_' + str(i)] = 0

                # collate and update metrics
                metrics_list = []
                for i in metrics.metric_names:
                    if i not in metrics_slug:
                        metrics_list.append(metrics.metrics[
                            metrics.metric_names.index(i)][0])
                    else:
                        metrics_list.append(metrics_slug[i])

                # update metrics
                metrics.update(metrics_list)

            print(metrics.get_stat_string(mode=0))
            print('invalids', len(invalids))
            logging.info("EVAL: metrics: {}".format(metrics.get_stat_string(mode=0)))
            logging.info("EVAL: invalids: {}".format(len(invalids)))

           # del h3d
            eval_loader.dataset._load_envs()
            t += 1
            if len(eval_loader.dataset.pruned_env_set) == 0:
                done = True

    elif 'lstm' in args.model_type:

        done = False
        t=0
        while done == False:

            if args.overfit:
                metrics = NavMetric(
                    info={'split': args.eval_split,
                          'thread': rank},
                    metric_names=[
                        'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                        'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                        'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                        'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                        'ep_len_30', 'ep_len_50'
                    ],
                    log_json=args.output_log_path)

            for num, batch in enumerate(tqdm(eval_loader)):

                model.load_state_dict(shared_model.state_dict())
                model.to(device)

                idx, questions, answer, _, actions_in, actions_out, action_lengths, _ = batch

                question_var = Variable(questions.to(device))
                metrics_slug = {}

                # evaluate at multiple initializations
                for i in [10, 30, 50]:

                    if action_lengths[0] - 1 - i < 0:
                        invalids.append([idx[0], i])
                        continue

                    h3d = eval_loader.dataset.episode_house

                    # forward through lstm till spawn
                    if len(eval_loader.dataset.episode_pos_queue[:-i]) > 0:
                        images = eval_loader.dataset.get_frames(
                            h3d,
                            eval_loader.dataset.episode_pos_queue[:-i],
                            preprocess=True)

                        raw_img_feats = eval_loader.dataset.cnn(
                            Variable(torch.FloatTensor(images).to(device)))

                        actions_in_pruned = actions_in[:, :action_lengths[0] -i]
                        actions_in_var = Variable(actions_in_pruned.to(device))
                        action_lengths_pruned = action_lengths.clone().fill_(action_lengths[0] - i)
                        img_feats_var = raw_img_feats.view(1, -1, 3200)

                        if '+q' in args.model_type:
                            scores, hidden = model(
                                img_feats_var, question_var,
                                actions_in_var,
                                action_lengths_pruned.cpu().numpy())
                        else:
                            scores, hidden = model(
                                img_feats_var, False, actions_in_var,
                                action_lengths_pruned.cpu().numpy())
                        try:
                            init_pos = eval_loader.dataset.episode_pos_queue[-i]
                        except:
                            invalids.append([idx[0], i])
                            continue

                        action_in = torch.LongTensor(1, 1).fill_(
                            actions_in[0,action_lengths[0] - i]).to(device)
                    else:
                        init_pos = eval_loader.dataset.episode_pos_queue[-i]
                        hidden = model.nav_rnn.init_hidden(1)
                        action_in = torch.LongTensor(1, 1).fill_(0).to(device)

                    h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                    init_dist_to_target = h3d.get_dist_to_target(
                        h3d.env.cam.pos)
                    if init_dist_to_target < 0:  # unreachable
                        invalids.append([idx[0], i])
                        continue

                    img = h3d.env.render()
                    img = torch.from_numpy(img.transpose(
                        2, 0, 1)).float() / 255.0

                    img_feat_var = eval_loader.dataset.cnn(
                        Variable(img.view(1, 3, 224, 224).to(device))).view(1, 1, 3200)

                    episode_length = 0
                    episode_done = True

                    dists_to_target, pos_queue, actions = [init_dist_to_target], [init_pos], []
                    actual_pos_queue = [(h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw)]
                    
                    cov_batch_i = []
                    for step in range(args.max_episode_length):

                        episode_length += 1

                        if '+q' in args.model_type:
                            scores, hidden = model(
                                img_feat_var,
                                question_var,
                                Variable(action_in),
                                False,
                                hidden=hidden,
                                step=True)
                        else:
                            scores, hidden = model(
                                img_feat_var,
                                False,
                                Variable(action_in),
                                False,
                                hidden=hidden,
                                step=True)

                        prob = F.softmax(scores, dim=1)

                        action = int(prob.max(1)[1].data.cpu().numpy()[0])

                        actions.append(action)

                        img, _, episode_done = h3d.step(action)

                        episode_done = episode_done or episode_length >= args.max_episode_length

                        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                        
                        img_feat_var = eval_loader.dataset.cnn(
                            Variable(img.view(1, 3, 224, 224)
                                     .to(device))).view(1, 1, 3200)

                        action_in = torch.LongTensor(1, 1).fill_(action + 1).to(device)

                        dists_to_target.append(h3d.get_dist_to_target(h3d.env.cam.pos))
                        
                        pos_queue.append([
                            h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                            h3d.env.cam.pos.z, h3d.env.cam.yaw
                        ])

                        # Saty: semantic maps here
                        target_obj = eval_loader.dataset.target_obj['fine_class']
                        img_semantic = h3d.env.render(mode='semantic')
                        cov = coverage(img_semantic, target_obj, semantic_classes)
                        cov_batch_i.append(cov)
                        
                        if episode_done:
                            cov_batch_i.append(target_obj)
                            if num not in coverage_log[t].keys():
                                coverage_log[t][num] = {}
                            coverage_log[t][num].update({i:cov_batch_i})    
                            break

                        # if episode_done == True:
                        #     break

                        actual_pos_queue.append([h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw])

                    # compute stats
                    metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                    metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                    metrics_slug['d_D_' + str(i)] = dists_to_target[0] - dists_to_target[-1]
                    metrics_slug['d_min_' + str(i)] = np.array(dists_to_target).min()
                    metrics_slug['ep_len_' + str(i)] = episode_length
                    
                    if action == 3:
                        metrics_slug['stop_' + str(i)] = 1
                    else:
                        metrics_slug['stop_' + str(i)] = 0
                    
                    inside_room = []
                    for p in pos_queue:
                        inside_room.append(
                            h3d.is_inside_room(
                                p, eval_loader.dataset.target_room))
                    if inside_room[-1] == True:
                        metrics_slug['r_T_' + str(i)] = 1
                    else:
                        metrics_slug['r_T_' + str(i)] = 0
                    if any([x == True for x in inside_room]) == True:
                        metrics_slug['r_e_' + str(i)] = 1
                    else:
                        metrics_slug['r_e_' + str(i)] = 0

                # collate and update metrics
                metrics_list = []
                for i in metrics.metric_names:
                    if i not in metrics_slug:
                        metrics_list.append(metrics.metrics[
                            metrics.metric_names.index(i)][0])
                    else:
                        metrics_list.append(metrics_slug[i])

                # update metrics
                metrics.update(metrics_list)

            
            print(metrics.get_stat_string(mode=0))
            print('invalids', len(invalids))
            logging.info("EVAL: init_steps: {} metrics: {}".format(i, metrics.get_stat_string(mode=0)))
            logging.info("EVAL: init_steps: {} invalids: {}".format(i, len(invalids)))

            # del h3d
            eval_loader.dataset._load_envs()
            t +=1 

            print("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
            logging.info("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
            # assert len(eval_loader.dataset.pruned_env_set) > 0
            if len(eval_loader.dataset.pruned_env_set) == 0:
                done = True

    elif 'pacman' in args.model_type:
        done = False
        t = 0
        while done == False:
            
            if args.overfit:
                metrics = NavMetric(
                    info={'split': args.eval_split,
                          'thread': rank},
                    metric_names=[
                        'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                        'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                        'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                        'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                        'ep_len_30', 'ep_len_50'
                    ],
                    log_json=args.output_log_path)
            #time_img = time.strftime("%m_%d_%H:%M")
             
            for num, batch in enumerate(tqdm(eval_loader)):

                model.load_state_dict(shared_model.state_dict())
                model.to(device)

                idx, question, answer, actions, action_length = batch
                metrics_slug = {}

                answeris = answer.item()

                h3d = eval_loader.dataset.episode_house

                # evaluate at multiple initializations
                video_dir = '../video/nav'
                video_dir = os.path.join(video_dir,
                                                   args.time_id + '_' + args.identifier)
                for i in [10, 30, 50]:
                    #Satyen suggests Himi changes ----> works
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    time_now = time.strftime("%m_%d_%H:%M")
                    if args.render:
                        video_name = '%s_video_%d_%d.avi' %(time_now, i, answeris)
                        video = cv2.VideoWriter(video_name, fourcc, 5, (224, 224))

                    if i > action_length[0]:
                        invalids.append([idx[0], i])
                        continue

                    question_var = Variable(question.to(device))

                    controller_step = False
                    planner_hidden = model.planner_nav_rnn.init_hidden(1)

                    # get hierarchical action history
                    (
                        planner_actions_in, planner_img_feats,
                        controller_step, controller_action_in,
                        controller_img_feats, init_pos,
                        controller_action_counter
                    ) = eval_loader.dataset.get_hierarchical_features_till_spawn(
                        actions[0, :action_length[0] + 1].numpy(), i, args.max_controller_actions 
                    )

                    #planner_actions_in_OH = Variable(oneHot(planner_actions_in, 4).to(device))
                    planner_actions_in_var = Variable(planner_actions_in.to(device))
                    planner_img_feats_var = Variable(planner_img_feats.to(device))
                
                    # forward planner till spawn to update hidden state
                    for step in range(planner_actions_in.size(0)):

                        planner_scores, planner_hidden = model.planner_step(
                            question_var, planner_img_feats_var[step].view(1,1,3200),
                            planner_actions_in_var[step].view(1,1),
                            planner_hidden
                        )

                    h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                    init_dist_to_target = h3d.get_dist_to_target(
                        h3d.env.cam.pos)

                    if init_dist_to_target < 0:  # unreachable
                        invalids.append([idx[0], i])
                        continue

                    dists_to_target, pos_queue, pred_actions = [init_dist_to_target], [init_pos], []
                    planner_actions, controller_actions = [], []

                    episode_length = 0
                    if args.max_controller_actions > 1:
                        controller_action_counter = controller_action_counter % args.max_controller_actions 
                        controller_action_counter = max(controller_action_counter - 1, 0)
                    else:
                        controller_action_counter = 0

                    first_step = True
                    first_step_is_controller = controller_step
                    planner_step = True
                    action = int(controller_action_in)
                    
                    cov_batch_i = []

                    for step in range(args.max_episode_length):
                        if not first_step:
                            img = torch.from_numpy(img.transpose(
                                2, 0, 1)).float() / 255.0
                            img_feat_var = eval_loader.dataset.cnn(
                                Variable(img.view(1, 3, 224,
                                                  224).to(device))).view(
                                                      1, 1, 3200)
                        else:
                            img_feat_var = Variable(controller_img_feats.to(device)).view(1, 1, 3200)

                        if not first_step or first_step_is_controller:
                            # query controller to continue or not
                            controller_action_in = Variable(torch.LongTensor(1, 1).fill_(action).to(device))
                            
                            # Satty: Conver to oneHot here
                            # controller_action_var = Variable(torch.zeros(3).to(device))
                            #controller_action_var[controller_action_in] = 1
                            # controller_action_in = Variable(oneHot(controller_action_, 3).to(device))
                            # print("Controller_actions_in", controller_action_var, controller_action_var.size())

                            controller_scores = model.controller_step(
                                img_feat_var, 
                                controller_action_in,
                                planner_hidden[0])
                            
                            prob = F.softmax(controller_scores, dim=1)
                            controller_action = int(prob.max(1)[1].data.cpu().numpy()[0])

                            if controller_action == 1 and controller_action_counter < args.max_controller_actions - 1:
                                controller_action_counter += 1
                                planner_step = False
                            else:
                                controller_action_counter = 0
                                planner_step = True
                                controller_action = 0
                        
                            controller_actions.append(controller_action)
                            first_step = False

                        if planner_step:
                            if not first_step:
                                #action_in = Variable(torch.zeros(1, 1, 4).to(device))
                                #action_in[0, 0, action + 1] = 1
                                action_in = torch.LongTensor(1, 1).fill_(action + 1).to(device)
                                # print("Action In:", action_in)
                                
                                planner_scores, planner_hidden = model.planner_step(
                                    question_var,
                                    img_feat_var,
                                    action_in,
                                    planner_hidden)

                            prob = F.softmax(planner_scores, dim=1)
                            action = int(prob.max(1)[1].data.cpu().numpy()[0])
                            planner_actions.append(action)

                        episode_done = action == 3 or episode_length >= args.max_episode_length

                        episode_length += 1
                        dists_to_target.append(
                            h3d.get_dist_to_target(h3d.env.cam.pos))

                        pos_queue.append([
                            h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                            h3d.env.cam.pos.z, h3d.env.cam.yaw
                        ])
                        
                        # Saty: semantic maps here
                        target_obj = eval_loader.dataset.target_obj['fine_class']
                        img_semantic = h3d.env.render(mode='semantic')
                        cov = coverage(img_semantic, target_obj, semantic_classes)
                        cov_batch_i.append(cov)
                        
                        if episode_done:
                            cov_batch_i.append(target_obj)
                            if num not in coverage_log[t].keys():
                                coverage_log[t][num] = {}
                            coverage_log[t][num].update({i:cov_batch_i})    
                            break

                        img, _, _ = h3d.step(action)
                        #cv2.imwrite('{}-{}-{}-{}.png'.format(num, i, episode_length, time_img), img)
                        if args.render:
                            # cv2.imshow('window', img)
                            # cv2.waitKey(100)
                            video.write(img)
                        first_step = False

                    if args.render:
                        video.release()
                    # compute stats
                    metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                    metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                    metrics_slug['d_D_' + str(
                        i)] = dists_to_target[0] - dists_to_target[-1]
                    metrics_slug['d_min_' + str(i)] = np.array(
                        dists_to_target).min()
                    metrics_slug['ep_len_' + str(i)] = episode_length
                    if action == 3:
                        metrics_slug['stop_' + str(i)] = 1
                    else:
                        metrics_slug['stop_' + str(i)] = 0
                    inside_room = []
                    for p in pos_queue:
                        inside_room.append(
                            h3d.is_inside_room(
                                p, eval_loader.dataset.target_room))
                    if inside_room[-1] == True:
                        metrics_slug['r_T_' + str(i)] = 1
                    else:
                        metrics_slug['r_T_' + str(i)] = 0
                    if any([x == True for x in inside_room]) == True:
                        metrics_slug['r_e_' + str(i)] = 1
                    else:
                        metrics_slug['r_e_' + str(i)] = 0

                # collate and update metrics
                metrics_list = []
                for i in metrics.metric_names:
                    if i not in metrics_slug:
                        metrics_list.append(metrics.metrics[
                            metrics.metric_names.index(i)][0])
                    else:
                        metrics_list.append(metrics_slug[i])

                # update metrics
                metrics.update(metrics_list)

            try:
                print(metrics.get_stat_string(mode=0))
                logging.info("EVAL: metrics: {}".format(metrics.get_stat_string(mode=0)))
            except:
                pass
            
            print('epoch', epoch)
            print('invalids', len(invalids))
            logging.info("EVAL: epoch {}".format(epoch))
            logging.info("EVAL: invalids {}".format(invalids))

            # del h3d
            eval_loader.dataset._load_envs()
            t +=1 
            
            if len(eval_loader.dataset.pruned_env_set) == 0:
                done = True

    # checkpoint if best val loss in terms of coverage
    cov_avg = avgCov(coverage_log)

    if metrics.metrics[8][0] > best_eval_acc or cov_avg > best_cov:  # d_D_50
        if metrics.metrics[8][0] > best_eval_acc:
            best_eval_acc = metrics.metrics[8][0]
        if cov_avg > best_cov:
            best_cov = cov_avg

        if args.to_log == 1:
            metrics.dump_log()
            log_file = os.path.join(args.checkpoint_dir, 'coverage_log_{}_{:.4f}.pkl'.format(epoch, cov_avg))
            with open(log_file, "wb") as file_:
                pkl.dump(coverage_log, file_)

            model_state = get_state(model)

            aad = dict(args.__dict__)
            ad = {}
            for i in aad:
                if i[0] != '_':
                    ad[i] = aad[i]

            checkpoint = {'args': ad, 'state': model_state, 'epoch': epoch}

            checkpoint_path = '%s/epoch_%d_d_D_50_acc-%.04f_cov-%.04f.pt' % (
                args.checkpoint_dir, epoch, best_eval_acc, cov_avg)
            
            print('Saving checkpoint to %s' % checkpoint_path)
            logging.info("EVAL: Saving checkpoint to {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)

    print('[best_eval_d_D_50:%.04f; best Coverage:%.04f]' % (best_eval_acc, best_cov))
    logging.info("EVAL: [best_eval_d_D_50:{0:.2f}]".format(best_eval_acc))

    # eval_loader.dataset._load_envs(start_idx=0, in_order=True)
    # Dump coverageLog as pkl file
    
    return best_eval_acc, best_cov

def train(rank, args, resume_epoch = 0):
    # torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    model_kwargs = {
        'question_vocab': load_vocab(args.vocab_json)
    }
    
    master = BottleMasterGRU(**model_kwargs)
    slave_kwargs = {}
    slave = BottleSlaveRnn(**slave_kwargs)

    lossFn = MaskedNLLCriterion().cuda()
    
    master_optim = torch.optim.Adamax(
        filter(lambda p: p.requires_grad, master.parameters()),
        lr=args.learning_rate)
    slave_optim = torch.optim.Adamax(
        filter(lambda p: p.requires_grad, slave.parameters()),
        lr=args.learning_rate)

    # Loader here!
    if args.checkpoint_path !=False:
        checkpoint = torch.load(args.checkpoint_path, map_location={'cuda': 'cpu'})
        print('Loading State_dict from: %s' % args.checkpoint_path)
        logging.info("Loading state_dict from: {}".format(args.checkpoint_path))
        shared.load_state_dict(checkpoint['state'])
        resume_epoch = checkpoint['epoch']

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.model_type,
        'num_frames': 5,
        'map_resolution': args.map_resolution,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)],
        'to_cache': args.to_cache,
        'overfit': args.overfit,
        'max_controller_actions': args.max_controller_actions,
        'max_actions': args.max_actions
    }

    args.output_log_path = os.path.join(args.log_dir,'train_' + str(rank) + '.json')

    metrics = NavMetric(
        info={'split': 'train',
              'thread': rank},
        metric_names=['master_loss', 'navigator_loss'],
        log_json=args.output_log_path)

    train_loader = EqaDataLoader(**train_loader_kwargs)

    print('train_loader has %d samples' % len(train_loader.dataset))
    logging.info('TRAIN: train loader has {} samples'.format(len(train_loader.dataset)))

    count, epoch = 0, resume_epoch
    # In case you need to load a model
    best_eval_acc = 0 if args.best_eval_acc==0 else args.best_eval_acc
    best_cov = 0 if args.best_cov == 0 else args.best_cov
    master.train()
    master.cuda()
    slave.train()
    slave.cuda()

    for epoch in range(resume_epoch, args.max_epochs):

        done = False
        all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()

        while done == False:

            for num, batch in enumerate(train_loader):

                count += 1


                idx, questions, _, img_feats, pos_queue, prev_actions,gt_actions,prev_action_lengths,mask_gt, mask_prev = batch
                
                img_feats_var = Variable(img_feats.cuda())
                questions_var = Variable(questions.cuda())
                pos_queue_var = Variable(pos_queue.cuda()).float()
                mask_gt_var = Variable(mask_gt.cuda())
                mask_prev_var = Variable(mask_prev.cuda())
                gt_actions_var = Variable(gt_actions.cuda())
                ############# Permute indices ###################################
                prev_action_lengths, perm_idx = prev_action_lengths.sort(0, descending=True)
                
                img_feats_var = img_feats_var[perm_idx]
                questions_var = questions_var[perm_idx]
                pos_queue_var = pos_queue_var[perm_idx]
                prev_actions = prev_actions[perm_idx]
                gt_actions = gt_actions[perm_idx]
                mask_prev_var = mask_prev_var[perm_idx]
                mask_gt_var = mask_gt_var[perm_idx]
                #################################################
                modules_probs, master_hidden, ques_feats = master(questions_var,
                                                    img_feats_var,
                                                    prev_actions, 
                                                    prev_action_lengths.cpu().numpy(), 
                                                    pos_queue_var)

                # import pdb;pdb.set_trace()
                master_hidden_T = Variable(torch.zeros(master_hidden.size(0), img_feats_var.size(1)-1,master_hidden.size(2)).cuda() )
                master_hidden_T[:,:master_hidden.size(1),:] = master_hidden.detach()
                # Exclude the last one
                action_scores, _ = slave(img_feats_var[:,:-1,:,:,:].contiguous(), 
                                        ques_feats.detach(), 
                                        prev_action_lengths.cpu().numpy()-1,
                                        master_hidden_T)
                
                # import pdb;pdb.set_trace()
                logProb_modules = F.softmax(modules_probs, dim=1)
                logProb_actions = F.softmax(action_scores, dim=1)

                gt_module = torch.zeros_like(prev_actions).cuda()
                gt_module[np.arange(prev_action_lengths.size(0)), prev_action_lengths.long()] = 1
                
                # TODO: Construct GT for modules
                master_loss = lossFn(logProb_modules,
                                     gt_module[:, :prev_action_lengths.max()].contiguous().view(-1,1),
                                     mask_gt_var[:, :prev_action_lengths.max()].contiguous().view(-1,1))
                
                # TODO: GEt gt actions for slave
                slave_loss = lossFn(logProb_actions, 
                    gt_actions_var[:, :prev_action_lengths.max()-1].contiguous().view(-1, 1),
                    mask_gt_var[:, :prev_action_lengths.max()-1].contiguous().view(-1, 1))
                

                # zero grad
                master_optim.zero_grad()
                slave_optim.zero_grad()
                # update metrics
                # metrics.update([loss.data[0]])
                # logging.info("TRAIN CNN loss: {:.6f}".format(loss.data[0]))

                master_loss.backward()
                master_optim.step()
                
                master_optim.zero_grad()
                slave_optim.zero_grad()
                slave_loss.backward()
                slave_optim.step()
                # if t % args.print_every == 0:
                #     print(metrics.get_stat_string())
                #     logging.info("TRAIN: metrics: {}".format(metrics.get_stat_string()))
                #     if args.to_log == 1:
                #         metrics.dump_log()

                print('[CHECK][Cache:%d][Total:%d]' %
                      (len(train_loader.dataset.img_data_cache),
                       len(train_loader.dataset.env_list)))
                logging.info('TRAIN: [CHECK][Cache:{}][Total:{}]'.format(
                    len(train_loader.dataset.img_data_cache), len(train_loader.dataset.env_list)))

            if all_envs_loaded == False:
                train_loader.dataset._load_envs(in_order=True)
                if len(train_loader.dataset.pruned_env_set) == 0:
                    done = True
                    if args.to_cache == False:
                        train_loader.dataset._load_envs(
                            start_idx=0, in_order=True)
            else:
                done = True

        # if epoch % args.eval_every ==0:
        #     best_eval_acc, best_cov = eval(rank,args,shared_model, best_eval_acc,best_cov, epoch)

        if epoch % args.save_every == 0:

            model_state = get_state(model)
            optimizer_state = optim.state_dict()

            aad = dict(args.__dict__)
            ad = {}
            for i in aad:
                if i[0] != '_':
                    ad[i] = aad[i]

            checkpoint = {'args': ad, 
                          'state': model_state, 
                          'epoch': epoch, 
                          'optimizer': optimizer_state}

            checkpoint_path = '%s/epoch_%d_thread_%d.pt' % (
                args.checkpoint_dir, epoch, rank)
            print('Saving checkpoint to %s' % checkpoint_path)
            logging.info("TRAIN: Saving checkpoint to {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-train_h5', default='utils/data/pruned_train_v2.h5')
    parser.add_argument('-val_h5', default='utils/data/pruned_val_v2.h5')
    parser.add_argument('-test_h5', default='utils/data/pruned_test_v2.h5')
    parser.add_argument('-data_json', default='utils/data/pruned_data_json_v2.json')
    parser.add_argument('-vocab_json', default='data/new_vocab.json')

    parser.add_argument(
        '-target_obj_conn_map_dir',
        default='/home/ubuntu/space/500')
    parser.add_argument('-map_resolution', default=500, type=int)

    parser.add_argument(
        '-mode',
        default='train+eval',
        type=str,
        choices=['train', 'eval', 'train+eval'])
    parser.add_argument('-eval_split', default='val', type=str)

    # model details
    parser.add_argument(
        '-model_type',
        default='bottle',
        choices=['cnn', 'cnn+q', 'lstm', 'lstm+q', 'lstm-mult+q', 'pacman', 'bottle'])
    parser.add_argument('-max_episode_length', default=100, type=int)
    parser.add_argument('-curriculum', default=0, type=int)

    # optim params
    parser.add_argument('-batch_size', default=20, type=int)
    parser.add_argument('-learning_rate', default=1e-3, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)
    parser.add_argument('-overfit', default=False, action='store_true')
    parser.add_argument('-render', default=False, action='store_true')
    # bookkeeping
    parser.add_argument('-print_every', default=1, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-save_every', default=5, type=int) #optional if you would like to save specific epochs as opposed to relying on the eval thread
    parser.add_argument('-identifier', default='pacman')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/05_06/nav/')
    parser.add_argument('-log_dir', default='logs/05_06/nav/')
    parser.add_argument('-to_log', default=1, type=int)
    parser.add_argument('-to_cache', action='store_true')
    parser.add_argument('-max_controller_actions', type=int, default=5)
    parser.add_argument('-max_actions', type=int)
    parser.add_argument('-best_eval_acc', type=float, default=0)
    parser.add_argument('-best_cov', type=float, default=0)
    
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

    #MAX_CONTROLLER_ACTIONS = args.max_controller_actions

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(filename=os.path.join(args.log_dir, "run_{}.log".format(
                                                str(datetime.now()).replace(' ', '_'))),
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')

    try:
       args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
       args.gpus = [int(x) for x in args.gpus]
    except KeyError:
       print("CPU not supported")
       logging.info("CPU not supported")
       exit()

    # args.gpus = [0]

    if args.checkpoint_path != False:

        print('Loading checkpoint from %s' % args.checkpoint_path)
        logging.info("Loading checkpoint from {}".format(args.checkpoint_path))

        args_to_keep = ['model_type']

        checkpoint = torch.load(args.checkpoint_path, map_location={
            'cuda:0': 'cpu'
        })

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)


    # if set to overfit; set eval_split to train
    if args.overfit == True:
        args.eval_split = 'train'

    print(args.__dict__)
    logging.info(args.__dict__)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)


    resume_epoch = 0
    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        logging.info("Loading params from checkpoint: {}".format(args.checkpoint_path))
        shared_model.load_state_dict(checkpoint['state'])
        resume_epoch = checkpoint['epoch']
    
    if args.mode == 'eval':

        eval(args.gpus[0], args, shared_model, 0, 0)

    elif args.mode == 'train':

        train(args.gpus[0], args, resume_epoch = resume_epoch)

    else:
        processes = []

        # Start the eval thread
        #p = mp.Process(target=eval, args=(0, args, shared_model))
        #p.start()
        #processes.append(p)

        # Start the training thread(s)
        for rank in range(1, args.num_processes + 1):
            # for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, resume_epoch))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
