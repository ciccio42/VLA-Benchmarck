import os
import json
import argparse
import glob
import numpy as np

BLACK_LIST = [] #[0,5,10,15]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from robosuite experiments.")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing result files.")
    parser.add_argument("--task_name", type=str, default="pick_place", help="Name of the task to analyze.")
    parser.add_argument("--change_obj_pos", type=str, default="False", help="Whether to change object positions.")
    args = parser.parse_args()

    result_folders = glob.glob(os.path.join(args.path, f"rollout_{args.task_name}_*_{args.change_obj_pos}_*"))

    if not result_folders:
        print("No result folders found in the specified path.")
        exit(1)

    result = dict()
    for folder in result_folders:
        run_number = folder.split(f'rollout_{args.task_name}_')[-1].split('_')[0]
        print(f"Processing folder: {folder} (Run {run_number})")
        result[run_number] = dict()
        
        json_files = glob.glob(os.path.join(folder, "*.json"))
        
        for i, file_path in enumerate(json_files):
            with open(file_path, 'r') as file:
                data = json.load(file)
                variation_id = int(data.get("variation_id", 0))
                
                if variation_id in BLACK_LIST:
                    print(f"Skipping variation_id {variation_id}")
                    continue

                # print(f"Results from {file_path}:")
                # print(json.dumps(data, indent=4))
                if len(result[run_number]) == 0:
                    for metric in data.keys():
                        result[run_number][metric] = []
                for metric, value in data.items():
                    result[run_number][metric].append(value)

    # Average the results for each run
    aggregated_results = dict()
    for run in result.keys():
        for metric, values in result[run].items():
            if metric not in aggregated_results:
                aggregated_results[metric] = []
            aggregated_results[metric].append(np.mean(values))

    # Average across runs
    # print(f"Success values {aggregated_results['success']} ")
    final_results = {metric: (round(np.mean(values), 3), round(np.std(values), 3)) for metric, values in aggregated_results.items()}

    print("Final Results:")
    for metric, (mean, std) in final_results.items():
        print(f"{metric}: Mean = {mean:.4f}, Std = {std:.4f}")
    # Save final results to a JSON file
    output_file = os.path.join(args.path, f"final_results_{args.task_name}_{args.change_obj_pos}.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)    
            