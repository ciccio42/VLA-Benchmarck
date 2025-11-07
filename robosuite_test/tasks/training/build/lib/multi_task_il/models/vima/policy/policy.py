from __future__ import annotations
import os
import torch
import torch.nn as nn
from tokenizers import AddedToken
from einops import rearrange, repeat

import multi_task_il.models.vima.nn as vnn
from multi_task_il.models.vima.utils import *
import numpy as np
import cv2
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from collections import defaultdict, deque

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[0.195, 0.255], [0.045, 0.105], [-0.105, -0.045], [-0.255, -0.195]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}


class Policy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        xf_n_layers: int,
        sattn_n_heads: int,
        xattn_n_heads: int,
        views: list,
        return_dist: bool = True,
        concat_state: bool = False,
        ckpt_path: str = None
    ):
        super().__init__()

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # overwrite default parameters
            embed_dim = ckpt['cfg']['embed_dim']
            xf_n_layers = ckpt['cfg']['xf_n_layers']
            sattn_n_heads = ckpt['cfg']['sattn_n_heads']
            xattn_n_heads = ckpt['cfg']['xattn_n_heads']

        self.embed_dim = embed_dim
        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=xf_n_layers,
            n_head=sattn_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )

        self.obj_encoder = vnn.ObjEncoder(
            transformer_emb_dim=embed_dim,
            views=views,
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
        )

        self.end_effector_encoder = vnn.Embedding(
            num_embeddings=2, embedding_dim=2)

        self.obs_fusion_layer = nn.Linear(
            self.obj_encoder.output_dim + 2, embed_dim)

        self.prompt_embedding = vnn.WordEmbedding()
        self.t5_prompt_encoder = vnn.T5PromptEncoder()
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )

        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )

        self._views = views
        self._return_dist = return_dist
        self._concat_state = concat_state

        # Load pre-trained weights for xattn
        if ckpt_path != None:
            state_dict = dict()
            for k, v in ckpt["state_dict"].items():
                if "xattn_gpt" in k or 'obj_encoder' in k or 'end_effector_encoder' in k or 't5_prompt_encoder' in k or 'prompt_obj_post_layer' in k or 'obs_fusion_layer' in k or 'prompt_embedding' in k:
                    state_dict[k.replace("policy.", "")] = v
            self.load_state_dict(state_dict, strict=False)

        # Action decoder
        self.action_decoder = vnn.ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose_position": [100, 100, 100],
                "pose_rotation": [50, 50, 50],
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        # Action encoder-decoder
        self.action_encoder = vnn.ActionEmbedding(
            output_dim=embed_dim,
            embed_dict={
                "pose_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=3,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=3,
                    hidden_dim=256,
                    hidden_depth=1,
                )
            },
        )

        self._n_discrete_x_bins = 100
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 100
        self._n_discrete_rot_bins = 50

        self.train(mode=False)
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total params module:', params)

    def forward_single_step(self, obs_token: torch.Tensor, obs_mask: torch.Tensor, action_token: torch.Tensor | None, prompt_token: torch.Tensor, prompt_token_mask: torch.Tensor):
        out = dict()

        # 3. Action Token Prediction
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        n_max_objs = obs_token.shape[-2]
        L = L_obs * n_max_objs + L_action

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(L, B, dtype=torch.bool,
                           device=obs_token.device)
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")
        for q in range(n_max_objs):
            tokens[q:: n_max_objs + 1] = obs_token[q::n_max_objs]
            masks[q:: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs:: n_max_objs + 1] = action_token

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(
            prompt_token_mask, dim=1) - 1

        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        predicted_action_token_t = tokens_out[n_max_objs -
                                              1:: n_max_objs + 1]

        predicted_action_token_t = predicted_action_token_t[-1].unsqueeze(
            0)

        # Action distribution
        dist_dict = self.forward_action_decoder(
            predicted_action_token_t)
        out['dist_dict'] = dist_dict

        # Create tensors with logits
        for k, v in dist_dict.items():
            # create a tensor containing the logits for each action component along the last dimension
            if k == "pose_position":
                for i, dist in enumerate(v._dists):
                    if i == 0:
                        position_logits_x_t = dist.logits
                    elif i == 1:
                        position_logits_y_t = dist.logits
                    else:
                        position_logits_z_t = dist.logits

            elif k == "pose_rotation":
                for i, dist in enumerate(v._dists):
                    if i == 0:
                        rotation_logits_r_t = dist.logits
                    elif i == 1:
                        rotation_logits_p_t = dist.logits
                    else:
                        rotation_logits_y_t = dist.logits
            elif k == "gripper_action":
                gripper_logits_t = v.logits

        position_logits_x_trajectory = position_logits_x_t
        position_logits_y_trajectory = position_logits_y_t
        position_logits_z_trajectory = position_logits_z_t

        rotation_logits_r_trajectory = rotation_logits_r_t
        rotation_logits_p_trajectory = rotation_logits_p_t
        rotation_logits_y_trajectory = rotation_logits_y_t

        # gripper_logits_trajectory = gripper_logits_t

        out['position_x_logits'] = position_logits_x_trajectory
        out['position_y_logits'] = position_logits_y_trajectory
        out['position_z_logits'] = position_logits_z_trajectory

        out['rotation_r_logits'] = rotation_logits_r_trajectory
        out['rotation_p_logits'] = rotation_logits_p_trajectory
        out['rotation_y_logits'] = rotation_logits_y_trajectory

        # out['gripper_logits'] = gripper_logits_trajectory

        return out

    def forward(
        self,
        input: object,
        mode: str = 'train'
    ):
        B, T, OBJ_NUM, C, W, H = input['obs']['objects']['cropped_img']['front'].shape

        out = dict()
        # 1. Forward prompt assembly
        # Dim: L_seq B Emb_size
        prompt_tokens, prompt_token_masks = self.forward_prompt_assembly(
            prompts=(input['prompt_token_type'],
                     input['word_batch'],
                     input['image_batch']))
        out['prompt_tokens'] = prompt_tokens
        out['prompt_token_masks'] = prompt_token_masks

        # 2. Forward obs token
        # Dim: B T Num_obj Obj_Emb_size
        obs_token_batch, obs_mask_batch = self.forward_obs_token(
            input['obs'])

        inference_cache = {}
        position_logits_x_trajectory = None
        position_logits_y_trajectory = None
        position_logits_z_trajectory = None
        rotation_logits_r_trajectory = None
        rotation_logits_p_trajectory = None
        rotation_logits_y_trajectory = None
        gripper_logits_trajectory = None

        # Compute action for step t
        for t in range(T-1):
            if t == 0:
                inference_cache["obs_tokens"] = deque([], maxlen=T)
                inference_cache["obs_masks"] = deque([], maxlen=T)
                inference_cache["action_tokens"] = deque([], maxlen=T)
                position_logits_x_t = None
                position_logits_y_t = None
                position_logits_z_t = None
                rotation_logits_r_t = None
                rotation_logits_p_t = None
                rotation_logits_y_t = None
                gripper_logits_t = None

            # Dim: 1 B Num_obj Obj_Emb_size
            obs_token_this_step = rearrange(torch.index_select(
                obs_token_batch, 1, torch.tensor(t).to(obs_token_batch.device)), 'B T O E -> T B O E')
            obs_mask_this_step = rearrange(torch.index_select(
                obs_mask_batch, 1, torch.tensor(t).to(obs_token_batch.device)), 'B T O -> T B O')

            # prepare history
            inference_cache["obs_tokens"].append(
                obs_token_this_step[0])  # B O E
            inference_cache["obs_masks"].append(obs_mask_this_step[0])
            max_objs = max(x.shape[1] for x in inference_cache["obs_tokens"])
            obs_tokens_to_forward, obs_masks_to_forward = [], []
            obs_tokens_this_env, obs_masks_this_env = [], []
            for idx in range(len(inference_cache["obs_tokens"])):
                obs_this_env_this_step = inference_cache["obs_tokens"][idx]
                obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
                required_pad = max_objs - obs_this_env_this_step.shape[1]
                obs_tokens_this_env.append(
                    obs_this_env_this_step
                )
                obs_masks_this_env.append(
                    obs_mask_this_env_this_step
                )

            obs_tokens_to_forward = any_stack(obs_tokens_this_env, dim=0)
            obs_masks_to_forward = any_stack(obs_masks_this_env, dim=0)

            if t == 0:
                action_tokens_to_forward = None
            else:
                action_tokens_to_forward = any_stack(
                    inference_cache["action_tokens"], dim=0)

            obs_token = obs_tokens_to_forward
            obs_mask = obs_masks_to_forward
            action_token = action_tokens_to_forward
            prompt_token = prompt_tokens
            prompt_token_mask = prompt_token_masks

            # 3. Action Token Prediction
            L_obs, B = obs_token.shape[:2]
            L_action = 0 if action_token is None else action_token.shape[0]
            n_max_objs = obs_token.shape[-2]
            L = L_obs * n_max_objs + L_action

            tokens = torch.empty(
                L, B, self.embed_dim, dtype=torch.float32, device=obs_token_batch.device
            )
            masks = torch.ones(L, B, dtype=torch.bool,
                               device=obs_token_batch.device)
            obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
            obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
            obs_token = rearrange(obs_token, "B L E -> L B E")
            obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
            obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
            obs_mask = rearrange(obs_mask, "B L -> L B")
            for q in range(n_max_objs):
                tokens[q:: n_max_objs + 1] = obs_token[q::n_max_objs]
                masks[q:: n_max_objs + 1] = obs_mask[q::n_max_objs]
            if action_token is not None:
                tokens[n_max_objs:: n_max_objs + 1] = action_token

            position_ids = torch.cumsum(masks, dim=0) - 1
            position_ids = position_ids.long()
            prompt_position_ids = torch.cumsum(
                prompt_token_mask, dim=1) - 1

            tokens_out = self.xattn_gpt(
                obs_action_tokens=tokens,
                prompt_tokens=prompt_token,
                prompt_mask=prompt_token_mask,
                obs_action_masks=masks.transpose(0, 1),
                obs_action_position_ids=position_ids.transpose(0, 1),
                prompt_position_ids=prompt_position_ids,
            )

            predicted_action_token_t = tokens_out[n_max_objs -
                                                  1:: n_max_objs + 1]

            predicted_action_token_t = predicted_action_token_t[-1].unsqueeze(
                0)

            # Action distribution
            dist_dict = self.forward_action_decoder(
                predicted_action_token_t)
            # Compute the action component class
            predicted_actions = dict()
            for k, v in dist_dict.items():
                predicted_actions[k] = torch.reshape(
                    v.mode(), (1, B, 1)) if k == 'gripper_action' else v.mode()
            # Compute the predicted action
            actions = self._de_discretize_actions(predicted_actions)
            out['predicted_actions'] = actions
            # During training forward the GT action to compute the action embedding
            actions_to_embed = None
            if mode == 'train':
                actions_to_embed = dict()
                # for each component create a tensor 1, B, E
                action_this_step = rearrange(torch.index_select(
                    input['actions'], 1, torch.tensor(t).to(obs_token_batch.device)), 'B T A -> T B A')
                actions_to_embed['pose_position'] = action_this_step[:, :, :3]
                actions_to_embed['pose_rotation'] = action_this_step[:, :, 3:6]
                # actions_to_embed['gripper_action'] = torch.reshape(
                #     action_this_step[:, :, 6], (1, B, 1))
            else:
                actions_to_embed = predicted_actions

            action_tokens = self.forward_action_token(
                actions_to_embed)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            inference_cache["action_tokens"].append(action_tokens)

            # Create tensors with logits
            for k, v in dist_dict.items():
                # create a tensor containing the logits for each action component along the last dimension
                if k == "pose_position":
                    for i, dist in enumerate(v._dists):
                        if i == 0:
                            position_logits_x_t = dist.logits
                        elif i == 1:
                            position_logits_y_t = dist.logits
                        else:
                            position_logits_z_t = dist.logits

                elif k == "pose_rotation":
                    for i, dist in enumerate(v._dists):
                        if i == 0:
                            rotation_logits_r_t = dist.logits
                        elif i == 1:
                            rotation_logits_p_t = dist.logits
                        else:
                            rotation_logits_y_t = dist.logits
                # elif k == "gripper_action":
                #     gripper_logits_t = v.logits

            if t == 0:
                position_logits_x_trajectory = position_logits_x_t
                position_logits_y_trajectory = position_logits_y_t
                position_logits_z_trajectory = position_logits_z_t

                rotation_logits_r_trajectory = rotation_logits_r_t
                rotation_logits_p_trajectory = rotation_logits_p_t
                rotation_logits_y_trajectory = rotation_logits_y_t

                # gripper_logits_trajectory = gripper_logits_t

            else:
                position_logits_x_trajectory = torch.cat(
                    (position_logits_x_trajectory, position_logits_x_t), 0)
                position_logits_y_trajectory = torch.cat(
                    (position_logits_y_trajectory, position_logits_y_t), 0)
                position_logits_z_trajectory = torch.cat(
                    (position_logits_z_trajectory, position_logits_z_t), 0)

                rotation_logits_r_trajectory = torch.cat(
                    (rotation_logits_r_trajectory, rotation_logits_r_t), 0)
                rotation_logits_p_trajectory = torch.cat(
                    (rotation_logits_p_trajectory, rotation_logits_p_t), 0)
                rotation_logits_y_trajectory = torch.cat(
                    (rotation_logits_y_trajectory, rotation_logits_y_t), 0)

                # gripper_logits_trajectory = torch.cat(
                #     (gripper_logits_trajectory, gripper_logits_t), 0)

        out['position_x_logits'] = position_logits_x_trajectory
        out['position_y_logits'] = position_logits_y_trajectory
        out['position_z_logits'] = position_logits_z_trajectory

        out['rotation_r_logits'] = rotation_logits_r_trajectory
        out['rotation_p_logits'] = rotation_logits_p_trajectory
        out['rotation_y_logits'] = rotation_logits_y_trajectory

        # out['gripper_logits'] = gripper_logits_trajectory

        return out

    def train(self, mode: bool = True):

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        self.training = mode

        for module in self.children():
            if isinstance(module, vnn.WordEmbedding) or isinstance(module, vnn.T5PromptEncoder):
                module.train(False)

        return self

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        batch_word_emb = self.prompt_embedding(word_batch)
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        n_max_objs = batch_image_emb.shape[-2]

        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += n_max_objs
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)

        prompt_tokens, prompt_masks = [], []
        for i, raw_prompt in enumerate(raw_prompts_token_type):
            word_ptr, img_ptr = 0, 0
            assembled_prompt = []
            assembled_mask = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[i][word_ptr])
                    word_ptr += 1
                    assembled_mask.append(True)
                elif item == 1:
                    obj_mask = any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[i][img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=assembled_prompt.device,
            )
            assembled_prompt = torch.cat(
                [assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        any_to_torch_tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                        torch.zeros(
                            num_padding,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)
        if self.t5_prompt_encoder is not None:
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):
        objects, ee = obs["objects"], obs["ee"]
        B, T, O, _, _, _ = objects['cropped_img']['front'].shape
        leading_dims = ee.shape[:2]

        objects = objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(**objects)
        img_feats = img_feats.reshape((B, T, O, img_feats.shape[-1]))
        obj_mask = {
            k: objects["mask"][k].reshape(B, T, -1) for k in objects["mask"]
        }

        ee_feats = self.end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(
            1,  img_feats.shape[-3], img_feats.shape[-2], 1)

        obs_feats = self.obs_fusion_layer(
            torch.cat([img_feats, ee_feats], dim=-1))

        obj_mask = any_concat([obj_mask[view]
                               for view in sorted(self._views)], dim=-1)
        return obs_feats, obj_mask

    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose_position"][..., 0] = (
            actions["pose_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose_position"][..., 1] = (
            actions["pose_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose_position"][..., 2] = (
            actions["pose_position"][..., 2] / self._n_discrete_z_bins
        )
        actions["pose_rotation"] = (
            actions["pose_rotation"] / self._n_discrete_rot_bins
        )

        return actions


if __name__ == '__main__':
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    from vima import create_policy_from_ckpt
    ckpt_path = "/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/vima/ckpt/92M.ckpt"
    policy = create_policy_from_ckpt(ckpt_path=ckpt_path, device='cuda:3')
