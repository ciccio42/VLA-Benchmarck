import argparse
import glob
import pickle as pkl
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import imageio
import debugpy
import sys
sys.path.append('../.')
sys.path.append("/home/rsofnc000/Multi-Task-LFD-Framework/repo/TinyVLA")
sys.path.append("/home/rsofnc000/Multi-Task-LFD-Framework/repo/TinyVLA/test/robosuite_test/robosuite")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_pkl', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--delta_001_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--30000_chkpt/rollout_pick_place")
    parser.add_argument('--output_dir', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--delta_001_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--30000_chkpt/rollout_pick_place/videos", help="Directory to save the videos")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for additional output")
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        

    pkl_files = glob.glob(f"{args.path_to_pkl}/*.pkl")
    os.makedirs(args.output_dir, exist_ok=True)

    for pkl_file in pkl_files:
        print(f"Loading {pkl_file}")
        with open(pkl_file, 'rb') as f:
            traj = pkl.load(f)

        task_description = traj[0]['obs']['task_description']
        first_img = traj[0]['obs']['camera_front_image']
        height, width, _ = first_img.shape

        video_name = os.path.join(args.output_dir, os.path.basename(pkl_file).replace('.pkl', '.mp4'))

        with imageio.get_writer(video_name, fps=10, codec='libx264') as writer:
            for t in range(len(traj) - 1):
                img = traj[t]['obs']['camera_front_image']
                img_pil = Image.fromarray(img)  # Keep as RGB for imageio
                draw = ImageDraw.Draw(img_pil)

                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()

                text_position = (10, height - 20)
                text_bg_padding = 4

                # Compute text bounding box
                bbox = draw.textbbox((0, 0), task_description, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                rect_start = (text_position[0] - text_bg_padding, text_position[1] - text_bg_padding)
                rect_end = (text_position[0] + text_width + text_bg_padding, text_position[1] + text_height + text_bg_padding)

                draw.rectangle([rect_start, rect_end], fill=(0, 0, 0))
                draw.text(text_position, task_description, fill=(0, 255, 0), font=font)

                writer.append_data(np.array(img_pil))

        print(f"Saved video to {video_name}")
