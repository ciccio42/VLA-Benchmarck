import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multi_task_il.models import get_model
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from multi_task_il.models.rep_modules import BYOLModule, ContrastiveModule
from multi_task_il.models.basic_embedding import TemporalPositionalEncoding
from einops import rearrange, repeat, parse_shape
from collections import OrderedDict
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


class _LSTMOneMany(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, forward_t):
        super(_LSTMOneMany, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        self.forward_t = forward_t
        self.output_dim = output_dim

        self.input_embedding = nn.Linear(in_features=input_dim,
                                         out_features=hidden_dim)
        # (batch_dim, seq_dim, feature_dim)
        self.reccurent = nn.GRU(output_dim,
                                hidden_dim,
                                layer_dim,
                                batch_first=True)
        self.out_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, ht=None, ct=None):

        # Initialize prediction tensor
        predictions = torch.zeros(
            x.size(0), self.forward_t, self.output_dim).to(x.device)

        for t in range(self.forward_t):
            if t == 0:
                # Initialize hidden state with "encoder" output
                # B x SEQ x HIDDEN_DIM
                hidden_state = F.relu(self.input_embedding(x))
                hidden_state = rearrange(hidden_state, 'B T H -> T B H')
                input = torch.zeros(
                    x.shape[0], x.shape[1], self.output_dim).to(x.get_device())
            else:
                hidden_state = hidden
                input = output[:, None, :]

            output, hidden = self.reccurent(input, hidden_state)

            output = self.out_linear(output[:, -1, :])
            predictions[:, t, :] = torch.clone(output)

        return predictions


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=True, sep_var=False, lstm=False, lstm_config=None):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures = n_mixtures
        self._dist_size = torch.Size((out_dim, n_mixtures))
        self._lstm = lstm

        if lstm_config is not None:
            self._forward_t = lstm_config.get('forward_t', 1)
        else:
            self._forward_t = 1
        if not lstm:
            self._mu = nn.Linear(
                in_dim, self._forward_t * out_dim * n_mixtures)
            self._logit_prob = nn.Linear(
                in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
            if const_var:
                ln_scale = torch.randn(
                    out_dim, dtype=torch.float32) / np.sqrt(out_dim)
                self.register_parameter(
                    '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
            if sep_var:
                ln_scale = torch.randn((out_dim, n_mixtures),
                                       dtype=torch.float32) / np.sqrt(out_dim)
                self.register_parameter(
                    '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
            if not (const_var or sep_var):
                self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)
        else:
            # Implementing DiscreteLogHead as LSTM
            self._mu = _LSTMOneMany(input_dim=in_dim,
                                    output_dim=out_dim * n_mixtures,
                                    hidden_dim=lstm_config.get(
                                        'hidden_dim', 128),
                                    layer_dim=lstm_config.get(
                                        'layer_dim', 1),
                                    forward_t=lstm_config.get(
                                        'forward_t', 1))
            self._logit_prob = _LSTMOneMany(input_dim=in_dim,
                                            output_dim=out_dim * n_mixtures,
                                            hidden_dim=lstm_config.get(
                                                'hidden_dim', 128),
                                            layer_dim=lstm_config.get(
                                                'layer_dim', 1),
                                            forward_t=lstm_config.get(
                                                'forward_t', 1))
            if const_var:
                ln_scale = torch.randn(
                    out_dim, dtype=torch.float32) / np.sqrt(out_dim)
                self.register_parameter(
                    '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
            if sep_var:
                ln_scale = torch.randn((out_dim, n_mixtures),
                                       dtype=torch.float32) / np.sqrt(out_dim)
                self.register_parameter(
                    '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
            if not (const_var or sep_var):
                self._ln_scale = _LSTMOneMany(input_dim=in_dim,
                                              output_dim=out_dim * n_mixtures,
                                              hidden_dim=lstm_config.get(
                                                  'hidden_dim', 128),
                                              layer_dim=lstm_config.get(
                                                  'layer_dim', 1),
                                              forward_t=lstm_config.get(
                                                  'forward_t', 1))

    def forward(self, x):  #  x has shape B T d
        B, T, _ = x.shape

        if isinstance(self._mu, _LSTMOneMany):
            if len(x.shape) == 3:
                x = rearrange(x, 'B T d -> (B T) d')
                x = x[:, None, :]
        if isinstance(self._mu, _LSTMOneMany):
            mu = self._mu(x)
            mu = rearrange(mu, '(B T) S (A N) -> (B T) S A N',
                           B=B,
                           T=T,
                           A=self._dist_size[0],
                           N=self._dist_size[1])
            mu = rearrange(mu, '(B T) S A N -> B T S A N',
                           B=B,
                           T=T)
        else:
            mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
            assert not torch.isnan(
                mu).any(), "_DiscreteLogHead mu contains nan"

        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape(
                (x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            if not self._lstm:
                if len(ln_scale.shape) == 1:
                    ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
                    # (1, 1, 8, 1) -> (B T, dist_size[0], dist_size[1]) i.e. each mixture has the **same** constant variance
                else:  # the sep_val case:
                    ln_scale = repeat(
                        ln_scale, 'out_d n_mix -> B T out_d n_mix', B=x.shape[0], T=x.shape[1])
            else:
                ln_scale = repeat(
                    ln_scale, 'out_d n_mix -> B S out_d n_mix', B=x.shape[0], S=self._forward_t)
                ln_scale = rearrange(ln_scale,
                                     '(B T) S out_d n_mix -> B T S out_d n_mix',
                                     B=B, T=T)
        if isinstance(self._logit_prob(x), _LSTMOneMany):
            logit_prob = self._logit_prob(x)
            logit_prob = rearrange(logit_prob, '(B T) S (A N) -> (B T) S A N',
                                   B=B,
                                   T=T,
                                   A=self._dist_size[0],
                                   N=self._dist_size[1])
            logit_prob = rearrange(logit_prob, '(B T) S A N -> B T S A N',
                                   B=B,
                                   T=T)
        else:
            logit_prob = self._logit_prob(x).reshape(
                mu.shape) if self._n_mixtures > 1 else torch.ones_like(mu)

        return (mu, ln_scale, logit_prob)


class VideoImitation(nn.Module):
    """ The imitation policy model  """

    def __init__(
        self,
        latent_dim,
        load_target_obj_detector=False,
        target_obj_detector_step=0,
        target_obj_detector_path=None,
        freeze_target_obj_detector=False,
        remove_class_layers=True,
        load_contrastive=True,
        load_inv=True,
        concat_target_obj_embedding=True,
        concat_bb=False,
        bb_sequence=1,
        height=120,
        width=160,
        demo_T=4,
        obs_T=6,
        dim_H=7,
        dim_W=12,
        action_cfg=None,
        attn_cfg=None,
        sdim=7,
        concat_state=False,
        atc_config=None,
        curl_config=None,
        concat_demo_head=False,
        concat_demo_act=False,
        demo_mean=0,
        byol_config=dict(),
        simclr_config=dict(),
    ):
        super().__init__()
        self._remove_class_layers = remove_class_layers
        self._concat_bb = concat_bb
        self._bb_sequence = bb_sequence
        self._concat_img_emb = action_cfg.get("concat_img_emb", True)
        self._concat_demo_emb = action_cfg.get("concat_demo_emb", True)

        self._object_detector = None
        self._target_obj_detector_path = target_obj_detector_path
        if load_target_obj_detector:
            self.load_target_obj_detector(target_obj_detector_path=target_obj_detector_path,
                                          target_obj_detector_step=target_obj_detector_step,
                                          )

        self._load_target_obj_detector = load_target_obj_detector
        self._freeze_target_obj_detector = freeze_target_obj_detector
        self._load_contrastive = load_contrastive
        self._load_inv = load_inv
        self._demo_T = demo_T
        self._obs_T = obs_T
        self._concat_state = concat_state
        self._concat_target_obj_embedding = concat_target_obj_embedding
        # action processing
        assert action_cfg.n_mixtures >= 1, "must predict at least one mixture!"
        self.concat_demo_head = concat_demo_head
        self.concat_demo_act = concat_demo_act

        in_dim = self._object_detector._agent_backone.task_embedding_dim*dim_H*dim_W
        self._linear_embed_img = nn.Sequential(
            nn.Linear(in_dim, attn_cfg.embed_hidden),
            nn.Dropout(attn_cfg.dropout), nn.ReLU(),
            nn.Linear(attn_cfg.embed_hidden, latent_dim))
        self._linear_embed_demo = nn.Sequential(
            nn.Linear(
                self._object_detector._agent_backone.task_embedding_dim, attn_cfg.embed_hidden),
            nn.Dropout(attn_cfg.dropout), nn.ReLU(),
            nn.Linear(attn_cfg.embed_hidden, latent_dim))

        if "KP" not in target_obj_detector_path:
            ac_in_dim = int(latent_dim + float(concat_demo_act)
                            * latent_dim + float(concat_bb) * 4 * self._bb_sequence + float(concat_state) * sdim)
        else:
            img_emb_dim = latent_dim

            demo_emb_dim = latent_dim

            ac_in_dim = int(img_emb_dim * float(self._concat_img_emb) + float(self._concat_demo_emb)
                            * demo_emb_dim + float(concat_bb) * 4 + float(concat_state) * sdim)

        self._picking_module = None
        self._placing_module = None
        self._picking_module_inv = None
        self._placing_module_inv = None

        if action_cfg.n_layers == 1:
            self._picking_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.out_dim), nn.ReLU())
            self._placing_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.out_dim), nn.ReLU())
        elif action_cfg.n_layers == 2:
            self._picking_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim,
                          action_cfg.out_dim), nn.ReLU()
            )
            self._placing_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim,
                          action_cfg.out_dim), nn.ReLU()
            )

        else:
            raise NotImplementedError

        head_in_dim = int(action_cfg.out_dim +
                          float(concat_demo_head) * latent_dim)

        self.adim = action_cfg.adim
        self.n_mixtures = action_cfg.n_mixtures

        self._action_dist_picking = _DiscreteLogHead(
            in_dim=head_in_dim,
            out_dim=action_cfg.adim,
            n_mixtures=action_cfg.n_mixtures,
            const_var=action_cfg.const_var,
            sep_var=action_cfg.sep_var,
            lstm=action_cfg.get('is_recurrent', False),
            lstm_config=action_cfg.get('lstm_config', None)
        )
        self._action_dist_placing = _DiscreteLogHead(
            in_dim=head_in_dim,
            out_dim=action_cfg.adim,
            n_mixtures=action_cfg.n_mixtures,
            const_var=action_cfg.const_var,
            sep_var=action_cfg.sep_var,
            lstm=action_cfg.get('is_recurrent', False),
            lstm_config=action_cfg.get('lstm_config', None)
        )
        self.demo_mean = demo_mean
        self.first_phase = True

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(self)
        print('Total params in Imitation module:', params)
        print("\n---- Complete model ----\n")
        # summary(self)

    def load_target_obj_detector(self, target_obj_detector_path=None, target_obj_detector_step=-1, gpu_id=0):
        conf_file = OmegaConf.load(os.path.join(
            target_obj_detector_path, "config.yaml"))
        self._object_detector = hydra.utils.instantiate(
            conf_file.policy)
        weights = torch.load(os.path.join(
            target_obj_detector_path,
            f"model_save-{target_obj_detector_step}.pt"),
            map_location=torch.device(gpu_id))
        self._object_detector.load_state_dict(weights)
        # self._object_detector.to("cuda:0")
        self._object_detector.eval()
        for p in self._object_detector.parameters():
            p.requires_grad = False

    def get_conv_layer_reference(self,  model=None):
        if model == None:
            return []

        model_children = list(model.children())
        conv_layers = []
        for child in model_children:
            if type(child) == nn.Conv2d:
                conv_layers.append(child)
            conv_layer_ret = self.get_conv_layer_reference(child)
            if len(conv_layer_ret) != 0:
                conv_layers = conv_layers + conv_layer_ret

        return conv_layers

    def set_conv_layer_reference(self,  model=None):
        self.conv_layer_ref = self.get_conv_layer_reference(model=model)
        print(self.conv_layer_ref)

    def _get_action_distribution(self, action_module, action_dist, bb, img_embed, states, demo_embed, first_phase):
        if self.concat_demo_act:  # for action model
            ac_in = img_embed
            if self._concat_demo_emb:
                if img_embed is not None:
                    ac_in = torch.cat((img_embed, demo_embed), dim=2)
                else:
                    ac_in = demo_embed
            if self._concat_bb:
                bb = rearrange(bb, 'B T O D -> B T (O D)')
                if not self._concat_demo_emb and not self._concat_img_emb:
                    ac_in = bb
                else:
                    ac_in = torch.cat((ac_in, bb), dim=2)

            ac_in = F.normalize(ac_in, dim=2).clamp(min=1e-10)

        ac_in = torch.cat((ac_in, states), 2) if self._concat_state else ac_in

        # predict behavior cloning distribution
        ac_pred = action_module(
            ac_in.type(torch.float32)).type(torch.float32)
        if self.concat_demo_head:
            ac_pred = torch.cat((ac_pred, demo_embed), dim=2)
            # maybe better to normalize here
            ac_pred = F.normalize(ac_pred, dim=2)

        mu_bc, scale_bc, logit_bc = action_dist(
            ac_pred)
        return mu_bc, scale_bc, logit_bc

    def get_action(self, embed_out, target_obj_embedding=None, bb=None, ret_dist=True, states=None, eval=False, first_phase=True):
        """directly modifies output dict to put action outputs inside"""
        out = dict()
        B, obs_T, _, _ = bb.shape
        # single-head case
        if embed_out is not None:
            demo_embed, img_embed = embed_out['demo_embed'], embed_out['img_embed']
            img_embed = rearrange(
                img_embed, "(B T) C -> B T C", B=B, T=obs_T)

        ac_in = None
        if self._concat_img_emb:
            if self._concat_target_obj_embedding and not eval:
                ac_in = img_embed[:, 1:, :]
            elif self._concat_target_obj_embedding and eval:
                ac_in = img_embed
            else:
                ac_in = img_embed

        if self._concat_demo_emb:
            demo_embed = repeat(demo_embed, 'B d -> B ob_T d', ob_T=obs_T)
        else:
            demo_embed = None
            img_embed = None

        if self._concat_target_obj_embedding and not eval:
            target_obj_embedding = target_obj_embedding.repeat(1, obs_T, 1)
            img_embed = img_embed[:, 1:, :]
            img_embed = torch.cat((img_embed, target_obj_embedding), dim=2)
        elif self._concat_target_obj_embedding and eval:
            target_obj_embedding = target_obj_embedding.repeat(1, obs_T, 1)
            img_embed = img_embed
            img_embed = torch.cat((img_embed, target_obj_embedding), dim=2)

        mu_bc = torch.zeros(
            (B, obs_T, self.adim, self.n_mixtures)).to(bb.get_device())
        scale_bc = torch.zeros(
            (B, obs_T, self.adim, self.n_mixtures)).to(bb.get_device())
        logit_bc = torch.zeros(
            (B, obs_T, self.adim, self.n_mixtures)).to(bb.get_device())
        if not eval:
            first_phase_indx = first_phase == True
            second_phase_indx = first_phase == False

            if torch.sum(first_phase_indx.int()) != 0:
                mu_picking, scale_picking, logit_picking = self._get_action_distribution(
                    action_module=self._picking_module,
                    action_dist=self._action_dist_picking,
                    bb=bb[first_phase_indx, :, 0, :][:, :, None, :],
                    img_embed=ac_in[first_phase_indx] if ac_in is not None else ac_in,
                    states=states[first_phase_indx],
                    demo_embed=demo_embed[first_phase_indx] if demo_embed is not None else None,
                    first_phase=True
                )
                mu_bc[first_phase_indx] = mu_picking
                scale_bc[first_phase_indx] = scale_picking
                logit_bc[first_phase_indx] = logit_picking
            if torch.sum(second_phase_indx.int()) != 0:
                mu_place, scale_place, logit_place = self._get_action_distribution(
                    action_module=self._placing_module,
                    action_dist=self._action_dist_placing,
                    bb=bb[second_phase_indx, :, 1, :][:, :, None, :],
                    img_embed=ac_in[second_phase_indx] if ac_in is not None else ac_in,
                    states=states[second_phase_indx],
                    demo_embed=demo_embed[second_phase_indx] if demo_embed is not None else None,
                    first_phase=False
                )
                mu_bc[second_phase_indx] = mu_place
                scale_bc[second_phase_indx] = scale_place
                logit_bc[second_phase_indx] = logit_place

        else:
            if first_phase:
                mu_picking, scale_picking, logit_picking = self._get_action_distribution(
                    action_module=self._picking_module,
                    action_dist=self._action_dist_picking,
                    bb=bb[:, :, 0, :][:, :, None, :],
                    img_embed=ac_in,
                    states=states,
                    demo_embed=demo_embed if demo_embed is not None else None,
                    first_phase=True)
                mu_bc = mu_picking
                scale_bc = scale_picking
                logit_bc = logit_picking
            else:
                mu_place, scale_place, logit_place = self._get_action_distribution(
                    action_module=self._placing_module,
                    action_dist=self._action_dist_placing,
                    bb=bb[:, :, 1, :][:, :, None, :],
                    img_embed=ac_in,
                    states=states,
                    demo_embed=demo_embed if demo_embed is not None else None,
                    first_phase=False
                )
                mu_bc = mu_place
                scale_bc = scale_place
                logit_bc = logit_place

        out['bc_distrib'] = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc) \
            if ret_dist else (mu_bc.type(torch.float32), scale_bc.type(torch.float32), logit_bc.type(torch.float32))
        out['demo_embed'] = demo_embed
        out['img_embed'] = img_embed
        # multi-head case? maybe register a name for each action head
        return out

    def _compute_target_obj_embedding(self, embed_out):
        # 1. Take embedding computed from the demo and the  agent
        # Take only the first frame for the agent scene
        demo_embed, img_embed = embed_out['demo_embed'], embed_out['img_embed'][:, 0, :][:, None, :]

        assert demo_embed.shape[1] == self._demo_T
        obs_T = img_embed.shape[1]
        if self.demo_mean:
            demo_embed = torch.mean(demo_embed, dim=1)
        else:
            # only take the last image, should alread be attended tho
            demo_embed = demo_embed[:, -1, :]
        demo_embed = repeat(demo_embed, 'B d -> B ob_T d', ob_T=obs_T)
        ac_in = torch.cat((img_embed, demo_embed), dim=2)
        ac_in = F.normalize(ac_in, dim=2)

        target_object_embedding = self._target_obj_embedding(ac_in)

        return target_object_embedding

    def forward(
        self,
        images,
        context,
        bb=None,
        gt_classes=None,
        predict_gt_bb=False,
        states=None,
        ret_dist=True,
        eval=False,
        images_cp=None,
        context_cp=None,
        actions=None,
        target_obj_embedding=None,
        compute_activation_map=False,
        first_phase=None,
        t=-1
    ):
        B, obs_T, _, height, width = images.shape
        demo_T = context.shape[1]

        if self._concat_bb and self._object_detector is None:
            predict_gt_bb = True

        embed_out = dict()
        if self._concat_bb and not predict_gt_bb:
            # run inference for target object detector
            model_input = dict()
            model_input['demo'] = context
            model_input['images'] = images
            model_input['gt_bb'] = bb
            model_input['gt_classes'] = gt_classes
            prediction = self._object_detector(model_input,
                                               inference=True)
            embed_out['demo_embed'] = self._linear_embed_demo(
                prediction['task_embedding'].detach().clone())
            conv_in = rearrange(
                prediction['feature_map'].detach().clone(), 'B C H W -> B (C H W)')
            embed_out['img_embed'] = self._linear_embed_img(conv_in)
            if len(prediction['classes_final']) == B*obs_T:
                predicted_bb_list = list()
                # check if there is a valid bounding box
                # Project bb over image
                # 1. Get the index with target class
                for indx in range(len(prediction['classes_final'])):
                    target_indx_flags = prediction['classes_final'][indx] == 1
                    place_indx_flags = torch.zeros((1, 1))
                    if "KP" in self._target_obj_detector_path:
                        place_indx_flags = prediction['classes_final'][indx] == 2

                    # get target object bb
                    if torch.sum((target_indx_flags == True).int()) != 0:
                        # 2. Get the confidence scores for the target predictions and the the max
                        target_max_score_indx = torch.argmax(
                            prediction['conf_scores_final'][indx][target_indx_flags])
                        max_score_target = prediction['conf_scores_final'][indx][target_indx_flags][target_max_score_indx]
                        # project bounding box
                        scale_factor = self._object_detector.get_scale_factors()
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][indx][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][target_indx_flags][target_max_score_indx][None, :]
                    else:
                        # print("No bb target")
                        # Get index for target object
                        predicted_bb = torch.zeros(
                            (1, 4)).to(device=images.get_device())

                    # get place bb
                    if torch.sum((place_indx_flags == True).int()) != 0 and "KP" in self._target_obj_detector_path:
                        # 2. Get the confidence scores for the target predictions and the the max
                        place_max_score_indx = torch.argmax(
                            prediction['conf_scores_final'][indx][place_indx_flags])
                        max_score_place = prediction['conf_scores_final'][indx][place_indx_flags][place_max_score_indx]
                        # project bounding box
                        scale_factor = self._object_detector.get_scale_factors()
                        predicted_bb_place = project_bboxes(bboxes=prediction['proposals'][indx][None][None],
                                                            width_scale_factor=scale_factor[0],
                                                            height_scale_factor=scale_factor[1],
                                                            mode='a2p')[0][place_indx_flags][place_max_score_indx][None, :]
                        predicted_bb = torch.concat(
                            (predicted_bb, predicted_bb_place))
                    elif "KP" in self._target_obj_detector_path:
                        # print("No bb place")
                        # Get index for target object
                        predicted_bb = torch.concat((predicted_bb, torch.zeros(
                            (1, 4)).to(device=images.get_device())))

                    predicted_bb_list.append(predicted_bb)

                predicted_bb = torch.stack(
                    predicted_bb_list, dim=0)
                predicted_bb = rearrange(
                    predicted_bb, "(B T) O D -> B T O D", B=B, T=obs_T)
            else:
                print("No bb for some frames")
                # Get index for target object
                target_index = gt_classes == 1
                predicted_bb = bb[target_index, :]

            predicted_bb = torch.clamp(predicted_bb, min=0.0)
            assert not torch.isnan(predicted_bb).any(
            ), "The tensor contains NaN values"
            assert (predicted_bb >= 0).all(
            ), "The tensor contains values less than zero"

        elif self._concat_bb and predict_gt_bb:
            if self._bb_sequence == 1:
                # get the target object
                B, T, O, D = bb.shape
                predicted_bb = bb
                if O == 2 and "KP" not in self._target_obj_detector_path:
                    predicted_bb = bb[:, :, 0, :]
                    predicted_bb = predicted_bb[:, :, None, :]
            else:
                # get the target object
                B, T, S, O, D = bb.shape
                predicted_bb = bb

        if self._concat_bb:
            out = self.get_action(
                embed_out=embed_out,
                target_obj_embedding=target_obj_embedding,
                bb=predicted_bb,
                ret_dist=ret_dist,
                states=states,
                eval=eval,
                first_phase=self.first_phase if eval else first_phase)
        else:
            out = self.get_action(
                embed_out=embed_out,
                target_obj_embedding=target_obj_embedding,
                bb=None,
                ret_dist=ret_dist,
                states=states,
                eval=eval)

        if self._concat_bb:
            out['predicted_bb'] = predicted_bb
            if not predict_gt_bb:
                out['target_obj_prediction'] = prediction

        # if self._concat_target_obj_embedding:
        #     out["target_obj_embedding"] = target_obj_embedding

        # if eval:
        #     return out  # NOTE: early return here to do less computation during test time

        return out

    def momentum_update(self, frac):
        self._byol.update_mom(frac)
        return

    def soft_param_update(self):
        self._byol.soft_param_update()
        self._simclr.soft_param_update()
        tau = 1 - self._byol.mom
        for param, target_param in zip(self._embed.parameters(), self._target_embed.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)
        return

    def freeze_attn_layers(self, n_layers=2):
        assert n_layers <= self._embed._attn_layers._n_layers, 'Attention only has %s layers' % self._embed._attn_layers._n_layers
        count = 0
        for n in range(n_layers):
            num_frozen = self._embed._attn_layers.freeze_layer(n)
            count += num_frozen
        print("Warning! Freeze the _First_ {} layers of attention! A total of {} params are frozen \n".format(
            n_layers, count))

    def freeze_img_encoder(self):
        count = 0
        for p in self._embed._img_encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                count += np.prod(p.shape)
        print("Warning! Freeze %s parameters in the image encoder \n" % count)

    def restart_action_layers(self):
        count = 0
        for module in [self._action_module, self._action_dist, self._inv_model]:
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                for p in layer.parameters():
                    count += np.prod(p.shape) if p.requires_grad else 0
        print("Re-intialized a total of %s parameters in action MLP layers" % count)

    def skip_for_eval(self):
        """ skip module inferences during evaluation"""
        self._byol = None
        self._simclr = None
        self._inv_model = None
        self._target_embed = None

    # def pretrain_img_encoder(self):
    #     """Freeze everything except for the image encoder + attention layers"""
    #     count = 0
    #     for p in zip(
    #         self._action_module.parameters(),
    #         self._inv_model.parameters(),
    #         self._embed._linear_embed.parameters()
    #         ):
    #         p.requires_grad = False
    #         count += np.prod(p.shape)
    #     print("Freezing action, inv, linear embed {} layer parameters".format(count))
