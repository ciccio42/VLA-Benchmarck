import pickle

if __name__ == "__main__":
    pkl_path = "/home/rsofnc000/checkpoint_save_folder/tiny_vla/tiny_vla_llava_pythia_lora_ur5e_pick_place_delta_removed_0_5_10_15_lora_r_128/dataset_stats.pkl"  # Replace with your actual file path
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(data.keys())
    
    print("Dataset Stats:")
    print(f"Action min {data['action_min']}")
    print(f"Action max {data['action_max']}")
    print(f"QPos mean {data['qpos_mean']}")
    print(f"QPos std {data['qpos_std']}")