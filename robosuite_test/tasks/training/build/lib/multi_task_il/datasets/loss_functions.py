import torch
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from torchvision.ops import box_iou
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes
from torchmetrics.classification import Accuracy
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce, repeat, parse_shape
from collections import defaultdict, OrderedDict
from torch.nn import CrossEntropyLoss
import copy

def calculate_maml_loss(config, device, meta_model, model_inputs):
    states, actions = model_inputs['states'], model_inputs['actions']
    images, context = model_inputs['images'], model_inputs['demo']
    aux = model_inputs['aux_pose']

    meta_model = meta_model.to(device)
    inner_iters = config.daml.get('inner_iters', 1)
    l2error = torch.nn.MSELoss()

    # error = 0
    bc_loss, aux_loss = [], []

    for task in range(states.shape[0]):
        learner = meta_model.clone()
        for _ in range(inner_iters):
            learner.adapt(
                learner(None, context[task], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = learner(states[task], images[task], ret_dist=False)
        l_aux = l2error(out['aux'], aux[task][None])[None]
        mu, sigma_inv, alpha = out['action_dist']
        action_distribution = DiscreteMixLogistic(
            mu[:-1], sigma_inv[:-1], alpha[:-1])
        l_bc = -torch.mean(action_distribution.log_prob(actions[task]))[None]
        bc_loss.append(l_bc)
        aux_loss.append(l_aux)

    return torch.cat(bc_loss, dim=0), torch.cat(aux_loss, dim=0)


def loss_func_bb(config, train_cfg, device, model, inputs, w_conf=1, w_reg=5, val=False):

    def compute_average_iou(gt_bb, pred_bb, batch_size):
        iou_t = box_iou(boxes1=torch.from_numpy(
            gt_bb), boxes2=pred_bb)

    def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
        target_pos = torch.ones_like(conf_scores_pos)
        target_neg = torch.zeros_like(conf_scores_neg)

        target = torch.cat((target_pos, target_neg))
        inputs = torch.cat((conf_scores_pos, conf_scores_neg))

        loss = F.binary_cross_entropy_with_logits(
            inputs, target, reduction='mean')

        return loss

    def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
        assert gt_offsets.size() == reg_offsets_pos.size()
        loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets,
                                reduction='mean')
        return loss

    def calc_classification_loss(cls_scores, gt_cls):
        # compute cross entropy loss
        # Prediction
        # [1, 0] -> no-target
        # [0, 1] -> target
        # GT-Target
        # 1 -> target
        # 0 -> no-target
        
        # compute sofmax on cls_scores and get the class with the highest probability
        # cls_cnt =  torch.sum(((torch.argmax(F.softmax(cls_scores, dim=-1), dim=-1) == 2) == True).int())
        
        
        cls_loss = F.cross_entropy(cls_scores, gt_cls.type(torch.int64))
        return cls_loss

    def compute_classification_accuracy(cls_scores, gt_cls):
        """Compute classification accuracy

        Args:
            cls_scores (torch.tensor): [N_positive_bb, 2] 
            gt_cls (torch.tensor): [N_positive_bb] target class
        """
        # create target from scores
        cls_prob = nn.Softmax(dim=-1)(cls_scores)
        accuracy = Accuracy(task="multiclass", num_classes=cls_prob.shape[1], top_k=1).to(
            cls_scores.get_device())
        return accuracy(cls_prob, gt_cls)

    def compute_avg_prec(gt_bb=None, predicted_bb=None, thr=0.7):
        from multi_task_il.models.cond_target_obj_detector.utils import get_iou_mat
        # compute IoU over time
        gt_bb = rearrange(gt_bb, 'B T N BB -> (B T N) BB')
        predicted_bb = rearrange(predicted_bb, 'B T N BB -> (B T N) BB')
        iou_t = torch.diagonal(
            box_iou(boxes1=gt_bb, boxes2=predicted_bb))  # (B T N) BB
        tp = (torch.where(iou_t > thr, 1.0, 0.0) == 1.0).sum(dim=0)
        return tp/gt_bb.shape[0]

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(inputs.items()):
        traj = inputs['traj']

        for key in traj.keys():
            model_inputs[key].append(traj[key].to(device))

        for key in inputs['demo_data'].keys():
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['images'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)

    all_losses = dict()

    # model = model.to(device)
    # cond_video = inputs['demo']
    # agent_obs = inputs['images']
    # gt_bb = inputs['gt_bb']
    # gt_classes = inputs['gt_classes']
    predictions_dict = model(inputs = [model_inputs['demo'], model_inputs['images'], model_inputs['gt_bb'], model_inputs['gt_classes']],
                             inference=val)

    if not val:
        # compute detection loss
        cls_loss = calc_cls_loss(predictions_dict['conf_scores_pos'],
                                 predictions_dict['conf_scores_neg'],
                                 traj['images'].shape[0]*traj['images'].shape[1])
        bb_reg_loss = calc_bbox_reg_loss(predictions_dict['GT_offsets'],
                                         predictions_dict['offsets_pos'],
                                         traj['images'].shape[0]*traj['images'].shape[1])

        # compute classification loss
        classification_loss = calc_classification_loss(predictions_dict['cls_scores'],
                                                       predictions_dict['GT_class_pos']
                                                       )

        all_losses["cls_loss"] = cls_loss
        all_losses["bb_reg_loss"] = bb_reg_loss
        all_losses["classification_loss"] = classification_loss
        all_losses["loss_sum"] = w_conf*cls_loss + \
            w_reg*bb_reg_loss + classification_loss

        # compute acccuracy
        class_accuracy = compute_classification_accuracy(
            cls_scores=predictions_dict['cls_scores'],
            gt_cls=predictions_dict['GT_class_pos'])
        all_losses["class_accuracy"] = class_accuracy
    else:
        pass

    # avg_iou = compute_average_iou(gt_bb=)

    # # compute average precision
    # proposals = predictions_dict['proposals'][:, None, None, :]
    # # take the bounding box with the highest confidence-score and compute the IoU with
    # scale_factor = model.get_scale_factors()
    # proposals = project_bboxes(bboxes=proposals,
    #                            width_scale_factor=scale_factor[0],
    #                            height_scale_factor=scale_factor[1],
    #                            mode='a2p')[:, None, :, :]
    # # all_losses["avg_prec"] = compute_avg_prec(gt_bb=model_inputs['gt_bb'],
    # #                                           predicted_bb=proposals)

    # if DEBUG:
    #     import cv2
    #     for indx in range(inputs['traj']['images'].shape[0]):
    #         image = np.array(np.moveaxis(
    #             inputs['traj']['images'][indx, 0, :, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
    #         proposal = proposals[indx].cpu()
    #         bb_gt = inputs['traj']['gt_bb'][indx][0][0].cpu()
    #         image = cv2.rectangle(np.ascontiguousarray(image),
    #                               (int(proposal[0, 0, 0]), int(
    #                                   proposal[0, 0, 1])),
    #                               (int(proposal[0, 0, 2]), int(
    #                                   proposal[0, 0, 3])),
    #                               color=(0, 0, 255), thickness=1)
    #         image = cv2.rectangle(np.ascontiguousarray(image),
    #                               (int(bb_gt[0]), int(
    #                                   bb_gt[1])),
    #                               (int(bb_gt[2]), int(
    #                                   bb_gt[3])),
    #                               color=(0, 255, 0), thickness=1)
    #         cv2.imwrite("prova_predictions_eval.png", image)

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
            else:
                task_losses[task_name][loss_name] = loss_val
    return task_losses


def calculate_obj_pos_loss(config, train_cfg, device, model, task_inputs, loss, accuracy):
    model_inputs = defaultdict(list)
    task_to_idx = dict()
    target_obj_pos_one_hot = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):

        for key in ['images', 'images_cp']:
            model_inputs[key].append(inputs['traj'][key].to(device))
        for key in ['demo', 'demo_cp']:
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_inputs[task_name]['traj']['target_position_one_hot'].requires_grad = True
        obj_position = task_inputs[task_name]['traj']['target_position_one_hot'].to(
            device)
        # obj_position.requires_grad = True
        target_obj_pos_one_hot[task_name] = obj_position
        task_bsize = inputs['traj']['images'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)

    all_losses = OrderedDict()
    out = model(
        images=model_inputs['images'], images_cp=model_inputs['images_cp'],
        context=model_inputs['demo'],  context_cp=model_inputs['demo_cp'])

    ##### ---- #####
    # ToDo generalize to multiple-task
    ##### ---- #####
    for task_name in target_obj_pos_one_hot.keys():
        all_losses[task_name] = dict()
        # for each task compute the cross-entropy loss
        # B - T - Number Classes
        gt = target_obj_pos_one_hot[task_name].permute(0, 2, 1)
        # gt.requires_grad = True
        prediction = out['target_obj_pred'].permute(0, 2, 1)
        all_losses[task_name]['ce_loss'] = loss(prediction, gt)

    all_accuracy = OrderedDict()
    for task_name in target_obj_pos_one_hot.keys():
        all_accuracy[task_name] = dict()
        gt = torch.argmax(
            target_obj_pos_one_hot[task_name].permute(0, 2, 1), dim=1)
        prediction = torch.argmax(
            out['target_obj_pred'].permute(0, 2, 1), dim=1)
        all_accuracy[task_name]['accuracy'] = accuracy(prediction, gt)

    return all_losses, all_accuracy


def loss_function_vima(config, train_cfg, device, model, task_inputs, mode='train'):
    model_inputs = defaultdict()
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['sample']
        input_keys = ['states',
                      'actions',
                      'prompt',
                      'prompt_token_type',
                      'word_batch',
                      'image_batch',
                      'obs']

        for key in input_keys:
            if key != 'prompt' and key != 'prompt_token_type':
                if key == 'image_batch' or key == 'obs':
                    model_inputs[key] = traj[key].to_torch_tensor(
                        device=device)
                else:
                    model_inputs[key] = traj[key].to(device)
            else:
                model_inputs[key] = traj[key]

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    # for key in model_inputs.keys():
    #     model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    out = model(
        input=model_inputs,
        mode=mode
    )

    # Compute Categorical-Cross-Entropy for each command component
    loss = CrossEntropyLoss(reduction="mean")
    for key in out.keys():
        if "position_x_logits" == key:
            prediction_x = rearrange(
                out['position_x_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32)
            x_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 0].to(torch.int64), out['position_x_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                x_true.requires_grad = True
            loss_x = loss(prediction_x, x_true.to(torch.float32))

        elif "position_y_logits" == key:
            y_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 1].to(torch.int64), out['position_y_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                y_true.requires_grad = True
            loss_y = loss(rearrange(
                out['position_y_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), y_true.to(torch.float32))

        elif "position_z_logits" == key:
            z_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 2].to(torch.int64), out['position_z_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                z_true.requires_grad = True
            loss_z = loss(rearrange(
                out['position_z_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), z_true.to(torch.float32))

        elif "rotation_r_logits" == key:
            r_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 3].to(torch.int64), out['rotation_r_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                r_true.requires_grad = True
            loss_r = loss(rearrange(
                out['rotation_r_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), r_true.to(torch.float32))

        elif "rotation_p_logits" == key:
            p_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 4].to(torch.int64), out['rotation_p_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                p_true.requires_grad = True
            loss_p = loss(rearrange(
                out['rotation_p_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), p_true.to(torch.float32))

        elif "rotation_y_logits" == key:
            yaw_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 5].to(torch.int64), out['rotation_y_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                yaw_true.requires_grad = True
            loss_yaw = loss(rearrange(
                out['rotation_y_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), yaw_true.to(torch.float32))

    all_losses['l_bc'] = loss_x + loss_y + \
        loss_z + loss_r + loss_p + loss_yaw  # + loss_gripper

    all_losses["loss_sum"] = all_losses["l_bc"]
    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            task_losses[task_name][loss_name] = torch.mean(loss_val)

    return task_losses


def calculate_task_loss(config, train_cfg, device, model, task_inputs, val=False):
    """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
        for each batch input, the model does only one forward pass."""

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['traj']
        input_keys = traj.keys()

        if config.get('use_daml', False):
            input_keys.append('aux_pose')
        for key in input_keys:
            model_inputs[key].append(traj[key].to(device))

        # if 'points' in traj.keys():
        #     model_inputs['points'].append(traj['points'].to(device).long())

        for key in inputs['demo_data'].keys():
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    if config.get('use_daml', False):
        bc_loss, aux_loss = calculate_maml_loss(
            config=config,
            device=device,
            meta_model=model,
            model_inputs=model_inputs)
        all_losses["l_bc"] = bc_loss
        all_losses["l_aux"] = aux_loss
        all_losses["loss_sum"] = bc_loss + aux_loss
    else:
        if "VideoImitation" in config.policy._target_:
            # assert model_inputs['images'].shape[0] == 12, f"Batch input {model_inputs['images'].shape[0]}"
            out = model(
                images=copy.deepcopy(model_inputs['images']),
                images_cp=copy.deepcopy(model_inputs['images_cp']),
                context=copy.deepcopy(model_inputs['demo']),
                context_cp=copy.deepcopy(model_inputs['demo_cp']),
                states=copy.deepcopy(model_inputs['states']),
                bb=copy.deepcopy(model_inputs['gt_bb']),
                gt_classes=copy.deepcopy(model_inputs['gt_classes']),
                ret_dist=False,
                actions=copy.deepcopy(model_inputs['actions']),
                first_phase=copy.deepcopy(model_inputs.get('first_phase', None)))
        elif "CondPolicy" in config.policy._target_:
            out = model(
                inputs=model_inputs,
                inference=False,
                oracle=False)
        else:  # other baselines
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'],
                ret_dist=False)

        # forward & backward action pred
        actions = model_inputs['actions']
        if "CondPolicy" not in config.policy._target_:
            mu_bc, scale_bc, logit_bc = out['bc_distrib']
            assert not torch.isnan(mu_bc).any(), "mu_bc contains nan"
            assert not torch.isnan(scale_bc).any(), "scale_bc contains nan"
            assert not torch.isnan(logit_bc).any(), "logit_bc contains nan"
            if "real" not in config.dataset_cfg.agent_name or ("real" in config.dataset_cfg.agent_name and config.dataset_cfg.get("pick_next", False)):
                # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 7
                action_distribution = DiscreteMixLogistic(
                    mu_bc[:, :-1], scale_bc[:, :-1], logit_bc[:, :-1])
            else:
                action_distribution = DiscreteMixLogistic(
                    mu_bc, scale_bc, logit_bc)

            act_prob = - action_distribution.log_prob(actions)
            if config.actions.get('is_recurrent', False):
                act_prob = rearrange(act_prob,
                                     'B T S A -> B (T S A)')
            else:
                act_prob = rearrange(act_prob,
                                     'B T A -> B (T A)')

        else:
            actions = rearrange(actions, 'B T act_dim -> (B T) act_dim')
            act_prob = - out['bc_distrib'].log_prob(actions)
            if len(act_prob.shape) == 1:
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])
            else:
                act_prob = torch.sum(act_prob, dim=-1)
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])

        assert not torch.isnan(act_prob).any(), "Act_prob contains nan"
        all_losses["l_bc"] = train_cfg.bc_loss_mult * \
            torch.mean(act_prob, dim=-1)

        if 'inverse_distrib' in out.keys():
            inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
            if "real" not in config.dataset_cfg.agent_name or ("real" in config.dataset_cfg.agent_name and config.dataset_cfg.get("pick_next", False)):
                inv_prob = - inv_distribution.log_prob(actions)
            else:
                inv_prob = - inv_distribution.log_prob(actions[:, :-1, :])

            if config.actions.get('is_recurrent', False):
                inv_prob = rearrange(inv_prob,
                                     'B T S A -> B (T S A)')
            else:
                inv_prob = rearrange(inv_prob,
                                     'B T A -> B (T A)')

            all_losses["l_inv"] = train_cfg.inv_loss_mult * \
                torch.mean(inv_prob, dim=-1)

        if 'point_ll' in out and train_cfg.pnt_loss_mult != 0.0:
            pnts = model_inputs['points']

            l_point = -train_cfg.pnt_loss_mult * out['point_ll'][range(pnts.shape[0]),
                                                                 pnts[:, -1,
                                                                      0].long(),
                                                                 pnts[:, -1, 1].long()]

            all_losses["point_loss"] = l_point

        # NOTE: the model should output calculated rep-learning loss
        if (hasattr(model, "_load_target_obj_detector") and hasattr(model, "_freeze_target_obj_detector")) or (hasattr(model.module, "_load_target_obj_detector") and hasattr(model.module, "_freeze_target_obj_detector")):
            try:
                if not model._load_target_obj_detector or not model._freeze_target_obj_detector:
                    rep_loss = torch.zeros_like(all_losses["l_bc"])
                    for k, v in out.items():
                        if k in train_cfg.rep_loss_muls.keys():
                            # just return size (B,) here
                            v = torch.mean(v, dim=-1)
                            v = v * train_cfg.rep_loss_muls.get(k, 0)
                            all_losses[k] = v
                            rep_loss = rep_loss + v
                    all_losses["rep_loss"] = rep_loss
                else:
                    all_losses["rep_loss"] = 0
            except:
                if not model.module._load_target_obj_detector or not model.module._freeze_target_obj_detector:
                    rep_loss = torch.zeros_like(all_losses["l_bc"])
                    for k, v in out.items():
                        if k in train_cfg.rep_loss_muls.keys():
                            # just return size (B,) here
                            v = torch.mean(v, dim=-1)
                            v = v * train_cfg.rep_loss_muls.get(k, 0)
                            all_losses[k] = v
                            rep_loss = rep_loss + v
                    all_losses["rep_loss"] = rep_loss
                else:
                    all_losses["rep_loss"] = 0
        else:
            pass

        loss_sum = 0
        for loss_key in ['l_bc', 'l_inv', 'rep_loss']:
            loss_sum += all_losses[loss_key] if loss_key in all_losses.keys() else 0.0
        all_losses["loss_sum"] = loss_sum

        all_losses["loss_sum"] = all_losses["loss_sum"] + \
            all_losses["point_loss"] if 'point_ll' in out else all_losses["loss_sum"]

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
    return task_losses


def calculate_grad_norm_loss(config, train_cfg, device, model, task_inputs):
    """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
    for each batch input, the model does only one forward pass."""

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['traj']
        input_keys = traj.keys()

        if config.get('use_daml', False):
            input_keys.append('aux_pose')
        for key in input_keys:
            model_inputs[key].append(traj[key].to(device))

        # if 'points' in traj.keys():
        #     model_inputs['points'].append(traj['points'].to(device).long())

        for key in inputs['demo_data'].keys():
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    if config.get('use_daml', False):
        bc_loss, aux_loss = calculate_maml_loss(
            config=config,
            device=device,
            meta_model=model,
            model_inputs=model_inputs)
        all_losses["l_bc"] = bc_loss
        all_losses["l_aux"] = aux_loss
        all_losses["loss_sum"] = bc_loss + aux_loss
    else:
        if config.policy._target_ == 'multi_task_il.models.mt_rep.VideoImitation':
            out = model(
                images=model_inputs['images'],
                images_cp=model_inputs['images_cp'],
                context=model_inputs['demo'],
                context_cp=model_inputs['demo_cp'],
                states=model_inputs['states'],
                bb=model_inputs['gt_bb'],
                gt_classes=model_inputs['gt_classes'],
                ret_dist=False,
                actions=model_inputs['actions'])
        elif "CondPolicy" in config.policy._target_:
            out = model(
                inputs=model_inputs,
                inference=False,
                oracle=False)
        else:  # other baselines
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'],
                ret_dist=False)

        # forward & backward action pred
        actions = model_inputs['actions']
        if "CondPolicy" not in config.policy._target_:
            if "real" not in config.dataset_cfg.agent_name:
                # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 7
                mu_bc, scale_bc, logit_bc = out['bc_distrib']
                action_distribution = DiscreteMixLogistic(
                    mu_bc[:, :-1], scale_bc[:, :-1], logit_bc[:, :-1])
                act_prob = rearrange(- action_distribution.log_prob(actions),
                                     'B n_mix act_dim -> B (n_mix act_dim)')
            else:
                mu_bc, scale_bc, logit_bc = out['bc_distrib']
                action_distribution = DiscreteMixLogistic(
                    mu_bc, scale_bc, logit_bc)
                act_prob = rearrange(- action_distribution.log_prob(actions),
                                     'B n_mix act_dim -> B (n_mix act_dim)')
        else:
            actions = rearrange(actions, 'B T act_dim -> (B T) act_dim')
            act_prob = - out['bc_distrib'].log_prob(actions)
            if len(act_prob.shape) == 1:
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])
            else:
                act_prob = torch.sum(act_prob, dim=-1)
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])

        all_losses["l_bc"] = train_cfg.bc_loss_mult * \
            torch.mean(act_prob, dim=-1)

        if 'inverse_distrib' in out.keys():
            if "real" not in config.dataset_cfg.agent_name:
                # compute inverse model density
                inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
                inv_prob = rearrange(- inv_distribution.log_prob(actions),
                                     'B n_mix act_dim -> B (n_mix act_dim)')
                all_losses["l_inv"] = train_cfg.inv_loss_mult * \
                    torch.mean(inv_prob, dim=-1)
            else:
                # compute inverse model density
                inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
                inv_prob = rearrange(- inv_distribution.log_prob(actions[:, :-1, :]),
                                     'B n_mix act_dim -> B (n_mix act_dim)')
                all_losses["l_inv"] = train_cfg.inv_loss_mult * \
                    torch.mean(inv_prob, dim=-1)

        if 'point_ll' in out:
            pnts = model_inputs['points']
            l_point = train_cfg.pnt_loss_mult * out['point_ll'][range(pnts.shape[0]),
                                                                pnts[:, -1, 0].long(), pnts[:, -1, 1].long()]

            all_losses["point_loss"] = l_point

        # NOTE: the model should output calculated rep-learning loss
        if (hasattr(model, "_load_target_obj_detector") and hasattr(model, "_freeze_target_obj_detector")) or (hasattr(model.module, "_load_target_obj_detector") and hasattr(model.module, "_freeze_target_obj_detector")):
            try:
                if not model._load_target_obj_detector or not model._freeze_target_obj_detector:
                    rep_loss = torch.zeros_like(all_losses["l_bc"])
                    for k, v in out.items():
                        if k in train_cfg.rep_loss_muls.keys():
                            # just return size (B,) here
                            v = torch.mean(v, dim=-1)
                            v = v * train_cfg.rep_loss_muls.get(k, 0)
                            all_losses[k] = v
                            rep_loss = rep_loss + v
                    all_losses["rep_loss"] = rep_loss
                else:
                    all_losses["rep_loss"] = 0
            except:
                if not model.module._load_target_obj_detector or not model.module._freeze_target_obj_detector:
                    rep_loss = torch.zeros_like(all_losses["l_bc"])
                    for k, v in out.items():
                        if k in train_cfg.rep_loss_muls.keys():
                            # just return size (B,) here
                            v = torch.mean(v, dim=-1)
                            v = v * train_cfg.rep_loss_muls.get(k, 0)
                            all_losses[k] = v
                            rep_loss = rep_loss + v
                    all_losses["rep_loss"] = rep_loss
                else:
                    all_losses["rep_loss"] = 0
        else:
            pass

        loss_sum = 0
        for loss_key in ['l_bc', 'l_inv', 'rep_loss']:
            loss_sum += all_losses[loss_key] if loss_key in all_losses.keys() else 0.0
        all_losses["loss_sum"] = loss_sum

        all_losses["loss_sum"] = all_losses["loss_sum"] + \
            all_losses["point_loss"] if 'point_ll' in out else all_losses["loss_sum"]

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
    return task_losses