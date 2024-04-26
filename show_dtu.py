import json
import numpy as np

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

output_dirs = ["exp_dtu/release"]

all_metrics = {"mean_d2s": [], "mean_s2d": [], "overall": []}
print(output_dirs)

for scene in scenes:
    print(scene,end=" ")
    for output in output_dirs:
        json_file = f"{output}/scan{scene}/test/ours_30000/tsdf/results.json"
        data = json.load(open(json_file))
        
        for k in ["mean_d2s", "mean_s2d", "overall"]:
            all_metrics[k].append(data[k])
            print(f"{data[k]:.3f}", end=" ")
        print()

latex = []
for k in ["mean_d2s", "mean_s2d", "overall"]:
    numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
    
    numbers = all_metrics[k] + [numbers]
    
    numbers = [f"{x:.2f}" for x in numbers]
    if k == "overall":
        latex.extend(numbers)
    
print(" & ".join(latex))