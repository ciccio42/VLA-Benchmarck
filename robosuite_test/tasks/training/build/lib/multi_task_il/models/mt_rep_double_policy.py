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
import cv2
from PIL import Image
from torchvision.transforms import ToPILImage

class _StackedAttnLayers(nn.Module):
    """
    Returns all intermediate-layer outputs, in case we want to re-use features + add more losses later
                -> default sharing norms now!
    Demo and obs share the same batchnorm at every layer;
    Construct a stack of layers at the same time, note this would give us more control 
    over how the attended embedding of demonstration and observation is combined.
    attention matrics here are even _smaller_ if we set fuse_starts to >0
    Args to note
    - fuse_starts: counts for how many layers demo & obs should attend to itself, independent of each other 

    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=3,
        fuse_starts=0,
        demo_ff_dim=128,
        obs_ff_dim=128,
        dropout=0,
        temperature=None,
        causal=False,
        n_heads=4,
        demo_T=4,
        compute_img_emb=True,
        compute_demo_emb=True
    ):
        super().__init__()
        assert demo_ff_dim % n_heads == 0, "n_heads must evenly divide feedforward_dim"
        self._n_heads = n_heads
        self._demo_ff_dim = demo_ff_dim
        self._obs_ff_dim = obs_ff_dim
        self._temperature = temperature if temperature is not None else np.sqrt(
            in_dim)
        self.compute_img_emb = compute_img_emb
        self.compute_demo_emb = compute_demo_emb

        self._obs_Qs, self._obs_Ks, self._obs_Vs = [
            nn.Sequential(*[nn.Conv3d(in_dim, obs_ff_dim, 1, bias=False)
                          for _ in range(n_layers)])
            for _ in range(3)]
        self._obs_Outs = nn.Sequential(
            *[nn.Conv3d(obs_ff_dim, out_dim, 1, bias=False) for _ in range(n_layers)])
        self._obs_a1s = nn.Sequential(
            *[nn.ReLU(inplace=dropout == 0) for _ in range(n_layers)])
        self._obs_drop1s = nn.Sequential(
            *[nn.Dropout3d(dropout) for _ in range(n_layers)])

        self._demo_Qs, self._demo_Ks, self._demo_Vs = [
            nn.Sequential(*[nn.Conv3d(in_dim, demo_ff_dim, 1, bias=False)
                          for _ in range(n_layers)])
            for _ in range(3)]
        self._demo_Outs = nn.Sequential(
            *[nn.Conv3d(demo_ff_dim, out_dim, 1, bias=False) for _ in range(n_layers)])
        self._demo_a1s = nn.Sequential(
            *[nn.ReLU(inplace=dropout == 0) for _ in range(n_layers)])
        self._demo_drop1s = nn.Sequential(
            *[nn.Dropout3d(dropout) for _ in range(n_layers)])
        self._norms = nn.Sequential(
            *[nn.BatchNorm3d(out_dim) for _ in range(n_layers)])

        self._n_layers = n_layers
        self._fuse_starts = fuse_starts
        print("A total of {} layers, cross demo-obs attention starts at layer idx {}".format(n_layers, fuse_starts))
        self._causal = causal  # keep causal only as an option for demo attention
        self._skip = out_dim == in_dim
        self._demo_T = demo_T

    def forward(self, inputs):
        """"""
        B, d, T, H, W = inputs.shape

        # obs_T could be as small as 1
        obs_T = T - self._demo_T
        out_dict = dict()
        for i in range(self._n_layers):
            demo_ly_in, obs_ly_in = inputs.split([self._demo_T, obs_T], dim=2)
            # -> (B, d, demo_T, H, W), (B, d, obs_T, H, W)
            if self.compute_demo_emb:
                # process demo first
                demo_q, demo_k, demo_v = [
                    rearrange(conv[i](
                        demo_ly_in), 'B (head ch) T H W -> B head ch (T H W)', head=self._n_heads)
                    for conv in [self._demo_Qs, self._demo_Ks, self._demo_Vs]
                ]
                # if self.query_task:
                #     demo_q = torch.cat((self.demo_q))
                a1, drop1 = [mod[i]
                             for mod in [self._demo_a1s, self._demo_drop1s]]
                B, head, ch, THW = demo_q.shape
                ff_dim = head * ch
                # B, heads, T_demo*HW, T_demo*HW
                demo_kq = torch.einsum(
                    'bnci,bncj->bnij', demo_k, demo_q) / self._temperature
                if self._causal:
                    mask = torch.tril(torch.ones(
                        (self._demo_T, self._demo_T))).to(demo_kq.device)
                    mask = mask.repeat_interleave(
                        H*W, 0).repeat_interleave(H*W, 1)  # -> (T*H*W, T*H*W)
                    # -> (1, 1, T*H*W, T*H*W)
                    demo_kq = demo_kq + \
                        torch.log(mask).unsqueeze(0).unsqueeze(0)
                demo_attn = F.softmax(demo_kq, 3)
                demo_v = torch.einsum('bncj,bnij->bnci', demo_v, demo_attn)
                demo_out = self._demo_Outs[i](
                    rearrange(demo_v, 'B head ch (T H W) -> B (head ch) T H W',
                              T=self._demo_T, H=H, W=W)
                )
                demo_out = demo_out + \
                    drop1(a1(demo_out)) if self._skip else drop1(demo_out)

            if self.compute_img_emb:
                # now, repeat demo's K and V for obs. NOTE: **brought the T dimension forward**
                obs_q, obs_k, obs_v = [
                    rearrange(conv[i](obs_ly_in),
                              'B (head ch) obs_T H W -> B obs_T head ch (H W)', head=self._n_heads)
                    for conv in [self._obs_Qs, self._obs_Ks, self._obs_Vs]
                ]
                a1, drop1 = [mod[i]
                             for mod in [self._obs_a1s, self._obs_drop1s]]
                if i >= self._fuse_starts:
                    rep_k, rep_v = [
                        repeat(rep, 'B head ch THW -> B obs_T head ch THW', obs_T=obs_T) for rep in [demo_k, demo_v]]
                    # only start attending to demonstration a few layers later
                    # now cat_k is B, T, head, ch, (4+1)HW
                    cat_k = torch.cat([rep_k, obs_k], dim=4)
                    cat_v = torch.cat([rep_v, obs_v], dim=4)
                else:
                    cat_k, cat_v = obs_k, obs_v  # only attend to observation selves
                obs_kq = torch.einsum(
                    'btnci,btncj->btnij', cat_k, obs_q) / self._temperature  # B, obs_T, heads, (1+T_demo)*HW, 1*HW
                assert obs_kq.shape[-2] == (1+self._demo_T) * H * W or \
                    obs_kq.shape[-2] == H*W
                # no causal mask is needed
                obs_attn = F.softmax(obs_kq, dim=4)

                obs_v = torch.einsum('btncj,btnji->btnci', cat_v, obs_attn)

                obs_out = self._obs_Outs[i](
                    rearrange(
                        obs_v, 'B T heads ch (H W) -> B (heads ch) T H W', H=H, W=W, T=obs_T)
                )
                obs_ly_in = obs_ly_in + \
                    drop1(a1(obs_out)) if self._skip else drop1(a1(obs_out))

            if self.compute_img_emb and self.compute_demo_emb:
                inputs = self._norms[i](
                    torch.cat([demo_ly_in, obs_ly_in], dim=2))
            elif self.compute_img_emb and not self.compute_demo_emb:
                inputs = self._norms[i](
                    obs_ly_in)
            elif not self.compute_img_emb and self.compute_demo_emb:
                inputs = self._norms[i](
                    demo_ly_in)

            out_dict['out_%s' % i] = inputs

        out_dict['last'] = inputs
        return out_dict

    def freeze_layer(self, i):
        count = 0
        for conv in [self._demo_Qs, self._demo_Ks, self._demo_Vs] + [self._obs_Qs, self._obs_Ks, self._obs_Vs]:
            to_freeze = conv[i]
            for param in to_freeze.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    count += np.prod(param.shape)
        for mod in [self._demo_a1s, self._demo_drop1s, self._obs_a1s, self._obs_drop1s, self._norms]:
            to_freeze = mod[i]
            for param in to_freeze.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    count += np.prod(param.shape)
        return count


class _TransformerFeatures(nn.Module):
    """
    Transformer-like module for computing self-attention on convolution features
    """

    def __init__(
            self, latent_dim, demo_T=4, dim_H=7, dim_W=12, embed_hidden=256, dropout=0.2, n_attn_layers=2, pos_enc=True, attn_heads=4, attn_ff=128, just_conv=False, pretrained=True, img_cfg=None, drop_dim=2, causal=True, attend_demo=True, demo_out=True, fuse_starts=0, concat_bb=False, compute_img_emb=True, compute_demo_emb=True, max_len=3000):
        super().__init__()

        flag, drop_dim = img_cfg.network_flag, img_cfg.drop_dim
        self.network_flag = flag
        self.compute_img_emb = compute_img_emb
        self.compute_demo_emb = compute_demo_emb

        assert flag == 0, "flag number %s not supported!" % flag
        self._img_encoder = get_model('resnet')(
            output_raw=True, drop_dim=drop_dim, use_resnet18=True, pretrained=img_cfg.pretrained)

        if drop_dim == 2:
            conv_feature_dim = 512
        elif drop_dim == 3:
            conv_feature_dim = 256
        else:
            conv_feature_dim = 128

        self._attn_layers = _StackedAttnLayers(
            in_dim=conv_feature_dim, out_dim=conv_feature_dim, n_layers=n_attn_layers,
            demo_ff_dim=attn_ff, obs_ff_dim=attn_ff, dropout=dropout,
            causal=causal, n_heads=attn_heads, demo_T=demo_T, fuse_starts=fuse_starts, compute_img_emb=compute_img_emb,
            compute_demo_emb=compute_demo_emb
        )

        self._pe = TemporalPositionalEncoding(
            conv_feature_dim, dropout, max_len=max_len) if pos_enc else None
        self.demo_out = demo_out
        in_dim = conv_feature_dim * dim_H * dim_W
        print("Linear embedder has input dim: {}x{}x{}={} ".format(
            conv_feature_dim, dim_H, dim_W, in_dim))

        self._linear_embed = nn.Sequential(
            nn.Linear(in_dim, embed_hidden),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(embed_hidden, latent_dim))

    def forward(self, images, context, compute_activation_map=False):
        assert len(
            images.shape) == 5, "expects [B, T, 3, height, width] tensor!"
        obs_T, demo_T = images.shape[1], context.shape[1]
        out_dict = OrderedDict()

        network_fn = self._resnet_features if self.network_flag == 0 else self._impala_features
        if self.compute_img_emb and self.compute_demo_emb:
            im_in = torch.cat((context, images), 1).float()
        elif self.compute_img_emb and not self.compute_demo_emb:
            im_in = images.float()
        elif not self.compute_img_emb and self.compute_demo_emb:
            im_in = context.float()

        im_features, no_pe_img_features = network_fn(im_in)
        out_dict['img_features'] = no_pe_img_features  # B T d H W
        out_dict['img_features_pe'] = rearrange(
            im_features, 'B d T H W -> B T d H W')
        # print(no_pe_img_features.shape)
        attn_out = self._attn_layers(im_features)
        # just use this for now, try other stuff later
        attn_features = attn_out['last']

        sizes = parse_shape(attn_features, 'B _ T _ _')
        features = rearrange(attn_features, 'B d T H W -> B T d H W', **sizes)
        out_dict['attn_features'] = features

        if self.compute_img_emb and self.compute_demo_emb:
            out_dict['demo_features'], out_dict['obs_features'] = \
                features.split([demo_T, obs_T], dim=1)
        elif self.compute_img_emb and not self.compute_demo_emb:
            out_dict['demo_features'] = None
            out_dict['obs_features'] = features
        elif not self.compute_img_emb and self.compute_demo_emb:
            out_dict['demo_features'] = features
            out_dict['obs_features'] = None

        # could also try do repre. on all intermediate layers too
        for k, v in attn_out.items():
            if k != 'last' and v.shape == attn_features.shape:
                reshaped = rearrange(v, 'B d T H W -> B T d H W', **sizes)
                out_dict['attn_'+k] = reshaped
                normalized = F.normalize(
                    self._linear_embed(rearrange(reshaped, 'B T d H W -> B T (d H W)')), dim=2)
                if self.compute_demo_emb and self.compute_img_emb:
                    out_dict['attn_'+k+'_demo'], out_dict['attn_'+k +
                                                          '_img'] = normalized.split([demo_T, obs_T], dim=1)
                elif self.compute_demo_emb and not self.compute_img_emb:
                    out_dict['attn_'+k+'_demo'] = normalized
                elif not self.compute_demo_emb and self.compute_img_emb:
                    out_dict['attn_'+k +
                             '_img'] = normalized
                # if True:
                #     demo_fm, img_fm = features.split([demo_T, obs_T], dim=1)
                #     import matplotlib.pyplot as plt
                #     # Squeeze the tensor to remove the batch dimension (1)
                #     img_fm = img_fm.squeeze()

                #     # Combine channels into a single heatmap
                #     heatmap = torch.sum(img_fm, dim=0)

                #     # Normalize the heatmap values between 0 and 1
                #     heatmap = (heatmap - heatmap.min()) / \
                #         (heatmap.max() - heatmap.min())

                #     # Convert the PyTorch tensor to a NumPy array for visualization
                #     heatmap_np = heatmap.numpy()

                #     # Plot the overlayed heatmap
                #     plt.imshow(heatmap_np, cmap='viridis')
                #     plt.colorbar()  # To add a colorbar for better understanding of the values
                #     plt.imsave("activation_map.png")

        out_dict['linear_embed'] = self._linear_embed(
            rearrange(features, 'B T d H W -> B T (d H W)'))
        normalized = F.normalize(out_dict['linear_embed'], dim=2)
        out_dict['normed_linear_embed'] = normalized

        if self.compute_img_emb and self.compute_demo_emb:
            demo_embed, img_embed = normalized.split([demo_T, obs_T], dim=1)
        elif self.compute_img_emb and not self.compute_demo_emb:
            img_embed = normalized
        elif not self.compute_img_emb and self.compute_demo_emb:
            demo_embed = normalized

        out_dict['demo_embed'] = demo_embed if self.compute_demo_emb else None
        out_dict['demo_mean'] = torch.mean(
            demo_embed, dim=1) if self.compute_demo_emb else None
        out_dict['img_embed'] = img_embed if self.compute_img_emb else None

        # NOTE(0427) this should always have length demo_T + obs_T now
        return out_dict

    def _resnet_features(self, x):
        if self._pe is None:
            return self._img_encoder(x)
        features = self._img_encoder(x)  # x is B, T, ch, h, w -> B, T, d, H, W
        pe_features = self._pe(features.transpose(1, 2))
        return pe_features, features  # B T d H W

    def _impala_features(self, x):
        # batch_size, concat_size = x.shape[0], x.shape[1] # B, T_c+T_im, 3, height, width
        sizes = parse_shape(x, 'B T _ _ _')
        x = rearrange(x, 'B T ch height width -> (B T) ch height width')
        features = self._img_encoder(x)  # B*T, d=256, H=6, W=9
        features = rearrange(features, '(B T) d H W -> B d T H W', **sizes)
        if self._pe is None:
            return features
        pe_features = self._pe(features)
        no_pe_features = rearrange(features, 'B d T H W -> B T d H W')

        return pe_features, no_pe_features


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
        zero_bb_after_pick=False,
        byol_config=dict(),
        simclr_config=dict(),
    ):
        super().__init__()
        self._remove_class_layers = remove_class_layers
        self._concat_bb = concat_bb
        self._bb_sequence = bb_sequence
        self._concat_img_emb = action_cfg.get("concat_img_emb", True)
        self._concat_demo_emb = action_cfg.get("concat_demo_emb", True)
        self._embed = None

        if self._concat_demo_emb or self._concat_img_emb:
            self._embed = _TransformerFeatures(
                latent_dim=latent_dim,
                demo_T=demo_T,
                dim_H=dim_H,
                dim_W=dim_W,
                concat_bb=concat_bb,
                compute_img_emb=action_cfg.get("concat_img_emb", True),
                compute_demo_emb=action_cfg.get("concat_demo_emb", True),
                **attn_cfg)

        self._object_detector = None
        self._target_obj_detector_path = target_obj_detector_path
        if load_target_obj_detector:
            self.load_target_obj_detector(target_obj_detector_path=target_obj_detector_path,
                                          target_obj_detector_step=target_obj_detector_step,
                                          )
        if load_contrastive:
            self._target_embed = copy.deepcopy(self._embed)
            self._target_embed.load_state_dict(self._embed.state_dict())
            for p in self._target_embed.parameters():
                p.requires_grad = False

            # one auxillary module calculate multiple losses
            # create a dummy input to calculate feature dimensions here
            with torch.no_grad():
                x = torch.zeros((1, demo_T+obs_T, 3, height, width))

                _out = self._embed(images=x[:, :demo_T], context=x[:, demo_T:])
                img_feats = _out['img_features']
                print("Image feature dimensions: {}".format(img_feats.shape))
                _, _, img_conv_dim, _, _ = img_feats.shape
                attn_feats = _out['attn_features']
                _, _, attn_conv_dim, _, _ = attn_feats.shape
                # should both be B, demo_T+obs, _, _, _
                assert img_feats.shape[1] == attn_feats.shape[1] or img_feats.shape[1] == attn_feats.shape[1] + demo_T
                img_feat_dim = np.prod(img_feats.shape[2:])  # should be d*H*W
                attn_feat_dim = np.prod(attn_feats.shape[2:])
                if img_feat_dim != attn_feat_dim:
                    print("Warning! pre and post attn features have different shapes:",
                          img_feat_dim, attn_feat_dim)

            self._byol = BYOLModule(
                embedder=self._target_embed,
                img_feat_dim=img_feat_dim, attn_feat_dim=attn_feat_dim,
                img_conv_dim=img_conv_dim, attn_conv_dim=attn_conv_dim, **byol_config)
            self._simclr = ContrastiveModule(
                embedder=self._target_embed,
                img_feat_dim=img_feat_dim, attn_feat_dim=attn_feat_dim,
                img_conv_dim=img_conv_dim, attn_conv_dim=attn_conv_dim, **simclr_config)

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
        assert not (
            concat_demo_head and concat_demo_act), 'Only support one concat type'
        print("Concat-ing embedded demo to action head? {}, to distribution head? {}".format(
            concat_demo_act, concat_demo_head))

        print(f"Concat state: {concat_state} - State dim {sdim}")
        if "KP" not in target_obj_detector_path:
            ac_in_dim = int(latent_dim + float(concat_demo_act)
                            * latent_dim + float(concat_bb) * 4 * self._bb_sequence + float(concat_state) * sdim)
        else:
            ac_in_dim = int(latent_dim * float(self._concat_img_emb) + float(self._concat_demo_emb)
                            * latent_dim + float(concat_bb) * 4 + float(concat_state) * sdim)

        inv_input_dim = int(2*ac_in_dim)

        self._picking_module = None
        self._placing_module = None
        self._picking_module_inv = None
        self._placing_module_inv = None

        if action_cfg.n_layers == 1:
            self._picking_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.out_dim), nn.ReLU())
            self._placing_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.out_dim), nn.ReLU())
            if load_inv:
                self._picking_module_inv = nn.Sequential(
                    nn.Linear(inv_input_dim, action_cfg.out_dim), nn.ReLU())
                self._placing_module_inv = nn.Sequential(
                    nn.Linear(inv_input_dim, action_cfg.out_dim), nn.ReLU())
        elif action_cfg.n_layers == 2:
            self._picking_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim, action_cfg.out_dim), nn.ReLU()
            )
            self._placing_module = nn.Sequential(
                nn.Linear(ac_in_dim, action_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(action_cfg.hidden_dim, action_cfg.out_dim), nn.ReLU()
            )
            if load_inv:
                self._picking_module_inv = nn.Sequential(
                    nn.Linear(inv_input_dim, action_cfg.hidden_dim), nn.ReLU(),
                    nn.Linear(action_cfg.hidden_dim,
                              action_cfg.out_dim), nn.ReLU()
                )
                self._placing_module_inv = nn.Sequential(
                    nn.Linear(inv_input_dim, action_cfg.hidden_dim), nn.ReLU(),
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

        if load_inv:
            self._action_dist_picking_inv = _DiscreteLogHead(
                in_dim=head_in_dim,
                out_dim=action_cfg.adim,
                n_mixtures=action_cfg.n_mixtures,
                const_var=action_cfg.const_var,
                sep_var=action_cfg.sep_var,
                lstm=action_cfg.get('is_recurrent', False),
                lstm_config=action_cfg.get('lstm_config', None)
            )
            self._action_dist_placing_inv = _DiscreteLogHead(
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

    def _load_model(self, model_path=None, step=0, conf_file=None, remove_class_layers=True, freeze=True):
        if model_path:
            # 1. Create the model starting from configuration
            model = hydra.utils.instantiate(conf_file.policy)
            summary(model)
            # 2. Load weights
            weights = torch.load(os.path.join(
                model_path, f"model_save-{step}.pt"), map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            # Remove classification layers
            if remove_class_layers:
                # Take the first _TransformerFeatures
                feature_extractor = list(model.children())[0]
                obj_classifiers = list(model.children())[-1]
                # do not take the classification layer
                target_obj_embedding_layers = list(
                    obj_classifiers.children())[:4]
                target_obj_embedding = nn.Sequential(
                    *target_obj_embedding_layers)
                for param in obj_classifiers.parameters():
                    param.requires_grad = False
                for param in target_obj_embedding.parameters():
                    param.requires_grad = False
            if freeze:
                for param in feature_extractor.parameters():
                    param.requires_grad = False

            return feature_extractor, obj_classifiers, target_obj_embedding
        else:
            raise ValueError("Model path cannot be None")

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

    def _get_inv_action_distribution(self, action_module, action_dist, img_embed, demo_embed, predicted_bb, states):
        inv_in = torch.cat((img_embed[:, :-1], img_embed[:, 1:]), 2)
        if self.concat_demo_act:
            if self._concat_bb:
                predicted_bb = rearrange(predicted_bb, 'B T O D -> B T (O D)')
                inv_in = torch.cat(
                    (
                        F.normalize(
                            torch.cat((img_embed[:, :-1], demo_embed[:, :-1], predicted_bb[:, :-1]), dim=2), dim=2),
                        F.normalize(
                            torch.cat((img_embed[:,  1:], demo_embed[:, :-1], predicted_bb[:, 1:]), dim=2), dim=2),
                    ),
                    dim=2)
            else:
                inv_in = torch.cat(
                    (
                        F.normalize(
                            torch.cat((img_embed[:, :-1], demo_embed[:, :-1]), dim=2), dim=2),
                        F.normalize(
                            torch.cat((img_embed[:,  1:], demo_embed[:, :-1]), dim=2), dim=2),
                    ),
                    dim=2)

            # print(inv_in.shape)
        if self._concat_state:
            inv_in = torch.cat(
                (torch.cat((inv_in, states[:, :-1]), dim=2), states[:, 1:]), dim=2)

        inv_pred = action_module(inv_in)

        if self.concat_demo_head:
            inv_pred = torch.cat((inv_pred, demo_embed[:, :-1]), dim=2)
            # maybe better to normalize here
            inv_pred = F.normalize(inv_pred, dim=2)

        mu_inv, scale_inv, logit_inv = action_dist(inv_pred)
        return mu_inv, scale_inv, logit_inv

    def _get_action_distribution(self, action_module, action_dist, bb, img_embed, states, demo_embed, first_phase):
        if self.concat_demo_act:  # for action model
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

        if self._concat_state:
            ac_in = torch.cat((ac_in, states), 2)

        ac_in = F.normalize(ac_in, dim=2)

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
        # single-head case
        if bb is not None:
            bb.requires_grad = True
        if embed_out is not None:
            demo_embed, img_embed = embed_out['demo_embed'], embed_out['img_embed']
            assert demo_embed.shape[1] == self._demo_T

        if not eval:
            obs_T = self._obs_T  # img_embed.shape[1]
        else:
            if img_embed is not None:
                obs_T = img_embed.shape[1]
            elif bb is not None:
                obs_T = bb.shape[1]

        ac_in = None
        if self._concat_img_emb:
            if self._concat_target_obj_embedding and not eval:
                ac_in = img_embed[:, 1:, :]
            elif self._concat_target_obj_embedding and eval:
                ac_in = img_embed
            else:
                ac_in = img_embed

        if self._concat_demo_emb:
            if self.demo_mean:
                demo_embed = torch.mean(demo_embed, dim=1)
            else:
                # only take the last image, should alread be attended tho
                demo_embed = demo_embed[:, -1, :]

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

        B, T, _, _ = bb.shape
        mu_bc = torch.zeros(
            (B, T, self.adim, self.n_mixtures)).to(bb.get_device())
        scale_bc = torch.zeros(
            (B, T, self.adim, self.n_mixtures)).to(bb.get_device())
        logit_bc = torch.zeros(
            (B, T, self.adim, self.n_mixtures)).to(bb.get_device())
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
        # if not eval:
        #     assert images_cp is not None, 'Must pass in augmented version of images'
        embed_out = None
        if self._concat_img_emb or self._concat_demo_emb:
            embed_out = self._embed(
                images, context, compute_activation_map=compute_activation_map)

        if self._concat_bb and self._object_detector is None:
            predict_gt_bb = True

        if self._concat_bb and not predict_gt_bb:
            # run inference for target object detector
            model_input = dict()
            model_input['demo'] = context
            model_input['images'] = images
            model_input['gt_bb'] = bb
            model_input['gt_classes'] = gt_classes
            self._object_detector.eval()
            prediction = self._object_detector(inputs=[context, images, bb, gt_classes],
                                               inference=True)
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
                        
                        # # plot predicted bb
                        # img = np.moveaxis(images[indx, 0].cpu().numpy()*255, 0, -1).astype(np.uint8)
                        # img = np.ascontiguousarray(img)
                        # img = cv2.rectangle(img, (int(predicted_bb[0][0].item()), int(predicted_bb[0][1].item())), (int(predicted_bb[0][2].item()), int(predicted_bb[0][3].item())), (0, 255, 0), 2)
                        # pil_image = Image.fromarray(img)
                        # pil_image.save(f"predicted_bb_{t}.png")
                        
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
                bb=torch.zeros(B, obs_T, 2, 4).to(images.get_device()),
                ret_dist=ret_dist,
                states=states,
                eval=eval,
                first_phase=self.first_phase if eval else first_phase)

        if self._concat_bb:
            out['predicted_bb'] = predicted_bb
            if not predict_gt_bb:
                out['target_obj_prediction'] = prediction

        if self._concat_target_obj_embedding:
            out["target_obj_embedding"] = target_obj_embedding

        if eval:
            return out  # NOTE: early return here to do less computation during test time

        if self._load_contrastive:
            # run frozen transformer on augmented images
            embed_out_target = self._target_embed(images_cp, context_cp)

            byol_out_dict = self._byol(embed_out, embed_out_target)
            for k, v in byol_out_dict.items():
                assert 'byol' in k
                out[k] = v

            simclr_out_dict = self._simclr(embed_out, embed_out_target)
            for k, v in simclr_out_dict.items():
                assert 'simclr' in k
                out[k] = v

        if embed_out is not None and out['demo_embed'] is not None:
            demo_embed, img_embed = out['demo_embed'], embed_out['img_embed']

        # B, T_im-1, d * 2
        if self._load_inv:
            first_phase_indx = first_phase == True
            second_phase_indx = first_phase == False

            B, T, _, _ = bb.shape
            mu_inv = torch.zeros(
                (B, T, self.adim, self.n_mixtures)).to(bb.get_device())
            scale_inv = torch.zeros(
                (B, T, self.adim, self.n_mixtures)).to(bb.get_device())
            logit_inv = torch.zeros(
                (B, T, self.adim, self.n_mixtures)).to(bb.get_device())

            if torch.sum(first_phase_indx.int()) != 0:
                mu_picking, scale_picking, logit_picking = self._get_inv_action_distribution(
                    action_module=self._picking_module_inv,
                    action_dist=self._action_dist_picking_inv,
                    img_embed=img_embed[first_phase_indx],
                    demo_embed=demo_embed[first_phase_indx],
                    predicted_bb=predicted_bb[first_phase_indx,
                                              :, 0, :][:, :, None, :],
                    states=states[first_phase_indx]
                )
                mu_inv[first_phase_indx] = mu_picking
                scale_inv[first_phase_indx] = scale_picking
                logit_inv[first_phase_indx] = logit_picking

            if torch.sum(second_phase_indx.int()) != 0:
                mu_place, scale_place, logit_place = self._get_inv_action_distribution(
                    action_module=self._placing_module_inv,
                    action_dist=self._action_dist_placing_inv,
                    img_embed=img_embed[second_phase_indx],
                    demo_embed=demo_embed[second_phase_indx],
                    predicted_bb=predicted_bb[second_phase_indx,
                                              :, 0, :][:, :, None, :],
                    states=states[second_phase_indx])
                mu_inv[second_phase_indx] = mu_place
                scale_inv[second_phase_indx] = scale_place
                logit_inv[second_phase_indx] = logit_place

            out['inverse_distrib'] = DiscreteMixLogistic(mu_inv, scale_inv, logit_inv) \
                if ret_dist else (mu_inv, scale_inv, logit_inv)

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
