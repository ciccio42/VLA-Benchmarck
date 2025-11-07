from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
import random
import albumentations as A
from collections import OrderedDict
import numpy as np
import cv2
from multi_task_il.datasets.utils import adjust_bb
from PIL import Image
DEBUG = False

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}

class DataAugmentation:
    
    def __init__(self, data_augs, mode, height, width, use_strong_augs, task_crops = OrderedDict(), agent_sim_crop = OrderedDict(), agent_crop = OrderedDict(), demo_crop = OrderedDict()):
        
        
        assert data_augs, 'Must give some basic data-aug parameters'
        if mode == 'train':
            print('Data aug parameters:', data_augs)
            
        self.data_augs = data_augs
        self.mode = mode
        self.height = height
        self.width = width
        self.use_strong_augs = use_strong_augs
        self.task_crops = task_crops
        self.agent_sim_crop = agent_sim_crop
        self.agent_crop = agent_crop
        self.demo_crop = demo_crop
        
        self.toTensor = ToTensor()
        old_aug = data_augs.get('old_aug', True)
        if old_aug:
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            jitters = {k: v * self.data_augs.get('weak_jitter', 0)
                    for k, v in JITTER_FACTORS.items()}
            weak_jitter = ColorJitter(**jitters)

            weak_scale = self.data_augs.get(
                'weak_crop_scale', (0.8, 1.0))
            weak_ratio = self.data_augs.get(
                'weak_crop_ratio', (1.6, 1.8))
            randcrop = RandomResizedCrop(
                size=(self.height, self.width), scale=weak_scale, ratio=weak_ratio)
            if self.data_augs.use_affine:
                randcrop = RandomAffine(degrees=0, translate=(self.data_augs.get(
                    'rand_trans', 0.1), self.data_augs.get('rand_trans', 0.1)))
            self.transforms = transforms.Compose([
                RandomApply([weak_jitter], p=0.1),
                RandomApply(
                    [GaussianBlur(kernel_size=5, sigma=self.data_augs.get('blur', (0.1, 2.0)))], p=0.1),
                randcrop,
                # self.normalize
            ])

            print("Using strong augmentations?", self.use_strong_augs)
            jitters = {k: v * self.data_augs.get('strong_jitter', 0)
                    for k, v in JITTER_FACTORS.items()}
            strong_jitter = ColorJitter(**jitters)
            self.grayscale = RandomGrayscale(
                self.data_augs.get("grayscale", 0))
            strong_scale = self.data_augs.get(
                'strong_crop_scale', (0.2, 0.76))
            strong_ratio = self.data_augs.get(
                'strong_crop_ratio', (1.2, 1.8))
            self.strong_augs = transforms.Compose([
                RandomApply([strong_jitter], p=0.05),
                self.grayscale,
                RandomHorizontalFlip(p=self.data_augs.get('flip', 0)),
                RandomApply(
                    [GaussianBlur(kernel_size=5, sigma=self.data_augs.get('blur', (0.1, 2.0)))], p=0.01),
                RandomResizedCrop(
                    size=(self.height, self.width), scale=strong_scale, ratio=strong_ratio),
                # self.normalize,
            ])
        else:
            # Imagenet-v1 normalization
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transforms = transforms.Compose([
                transforms.ColorJitter(
                    brightness=list(self.data_augs.get(
                        "brightness", [0.875, 1.125])),
                    contrast=list(self.data_augs.get(
                        "contrast", [0.5, 1.5])),
                    saturation=list(self.data_augs.get(
                        "contrast", [0.5, 1.5])),
                    hue=list(self.data_augs.get("hue", [-0.05, 0.05])),
                )
            ])
            print("Using strong augmentations?", self.use_strong_augs)
            self.strong_augs = transforms.Compose([
                transforms.ColorJitter(
                    brightness=list(self.data_augs.get(
                        "brightness_strong", [0.875, 1.125])),
                    contrast=list(self.data_augs.get(
                        "contrast_strong", [0.5, 1.5])),
                    saturation=list(self.data_augs.get(
                        "contrast_strong", [0.5, 1.5])),
                    hue=list(self.data_augs.get(
                        "hue_strong", [-0.05, 0.05]))
                ),
            ])

            self.affine_transform = A.Compose([
                # A.Rotate(limit=(-angle, angle), p=1),
                A.ShiftScaleRotate(shift_limit=0.1,
                                rotate_limit=0,
                                scale_limit=0,
                                p=self.data_augs.get(
                                    "p", 9.0))
            ])
        

    def horizontal_flip(self, obs, bb=None, p=0.1):
        if random.random() < p:
            height, width = obs.shape[-2:]
            obs = obs.flip(-1)
            if bb is not None:
                # For each bounding box
                for obj_indx, obj_bb in enumerate(bb):
                    x1, y1, x2, y2 = obj_bb
                    x1_new = width - x2
                    x2_new = width - x1
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1_new, y1, x2_new, y2]])
        return obs, bb

    def __call__(self, task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False, sim_crop=False):

        if perform_scale_resize:
            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            if len(self.demo_crop) != 0 and not agent:
                crop_params = self.demo_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.agent_crop) != 0 and agent and not sim_crop:
                crop_params = self.agent_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.agent_sim_crop) != 0 and agent and sim_crop:
                crop_params = self.agent_sim_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.task_crops) != 0:
                crop_params = self.task_crops.get(
                    task_name, [0, 0, 0, 0])

            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            
            # cv2.imwrite('obs_before_tensore.png', obs)
            # obs_pil = Image.fromarray(obs)
            # obs_pil.save(f"prova_resized_pil_{frame_number}.png")
            
            # obs = obs[:,:,::-1].copy()
            obs = obs.copy()
            
            # obs_pil = Image.fromarray(obs)
            # obs_pil.save(f"obs_before_tensor_{frame_number}.png")
            
            obs = self.toTensor(obs)
            
            # ---- Resized crop ----#
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(self.height, self.width))
            if DEBUG:
                cv2.imwrite(f"prova_resized_{frame_number}.png", np.moveaxis(
                    obs.numpy()*255, 0, -1))
            if bb is not None and class_frame is not None:
                bb = adjust_bb(dataset_loader=self,
                               bb=bb,
                               obs=obs,
                               img_height=img_height,
                               img_width=img_width,
                               top=top,
                               left=left,
                               box_w=box_w,
                               box_h=box_h)

            if self.data_augs.get('null_bb', False) and bb is not None:
                bb[0][0] = 0.0
                bb[0][1] = 0.0
                bb[0][2] = 0.0
                bb[0][3] = 0.0
        else:
            obs = self.toTensor(obs)
            if bb is not None and class_frame is not None:
                for obj_indx, obj_bb in enumerate(bb):
                    # Convert normalized bounding box coordinates to actual coordinates
                    x1, y1, x2, y2 = obj_bb
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1, y1, x2, y2]])

        # ---- Affine Transformation ----#
        if self.data_augs.get('affine', False) and agent:
            obs_to_affine = np.array(np.moveaxis(
                obs.numpy()*255, 0, -1), dtype=np.uint8)
            norm_bb = A.augmentations.bbox_utils.normalize_bboxes(
                bb, obs_to_affine.shape[0], obs_to_affine.shape[1])

            transformed = self.affine_transform(
                image=obs_to_affine,
                bboxes=norm_bb)
            obs = self.toTensor(transformed['image'])
            bb_denorm = np.array(A.augmentations.bbox_utils.denormalize_bboxes(bboxes=transformed['bboxes'],
                                                                               rows=obs_to_affine.shape[0],
                                                                               cols=obs_to_affine.shape[1]
                                                                               ))
            for obj_indx, obj_bb in enumerate(bb_denorm):
                if bb_denorm[obj_indx][0] > obs_to_affine.shape[1]:
                    bb_denorm[obj_indx][0] = obs_to_affine.shape[1]
                if bb_denorm[obj_indx][1] > obs_to_affine.shape[0]:
                    bb_denorm[obj_indx][1] = obs_to_affine.shape[0]
                if bb_denorm[obj_indx][2] > obs_to_affine.shape[1]:
                    bb_denorm[obj_indx][2] = obs_to_affine.shape[1]
                if bb_denorm[obj_indx][3] > obs_to_affine.shape[0]:
                    bb_denorm[obj_indx][3] = obs_to_affine.shape[0]
            bb = bb_denorm
        # ---- Augmentation ----#
        if self.use_strong_augs and second:
            augmented = self.strong_augs(obs)
            if DEBUG:
                cv2.imwrite("strong_augmented.png", np.moveaxis(
                    augmented.numpy()*255, 0, -1))
        else:
            if perform_aug:
                augmented = self.transforms(obs)
            else:
                augmented = obs
            if DEBUG:
                if agent:
                    cv2.imwrite("weak_augmented.png", np.moveaxis(
                        augmented.numpy()*255, 0, -1))
        assert augmented.shape == obs.shape

        if bb is not None:
            if False:
                image = np.ascontiguousarray(np.array(np.moveaxis(
                    augmented.numpy()*255, 0, -1), dtype=np.uint8))
                for single_bb in bb:
                    try:
                        image = cv2.rectangle(image,
                                              (int(single_bb[0]),
                                               int(single_bb[1])),
                                              (int(single_bb[2]),
                                               int(single_bb[3])),
                                              color=(0, 0, 255),
                                              thickness=1)
                    except:
                        print("Exception")
                cv2.imwrite("bb_cropped_after_aug.png", image)
            
            if self.height == 224 and self.width == self.width:
                augmented = self.normalize(augmented)
            
            # obs_pil = np.moveaxis(augmented.numpy()*255, 0, -1).astype(np.uint8)
            # obs_pil = Image.fromarray(obs_pil)
            # obs_pil.save(f"agent_augmented.png")
            
            return augmented, bb, class_frame
        else:
            if self.height == 224 and self.width == self.width:
                augmented = self.normalize(augmented)
            
            # obs_pil = np.moveaxis(augmented.numpy()*255, 0, -1).astype(np.uint8)
            # obs_pil = Image.fromarray(obs_pil)
            # obs_pil.save(f"augmented.png")
            
            return augmented