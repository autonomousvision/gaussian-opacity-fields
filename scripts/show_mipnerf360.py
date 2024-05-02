import json
import numpy as np

scenes = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']

# outdoor scenes
# scenes = scenes[:5]
# indoor scenes
# scenes = scenes[5:]

output_dirs = ["exp_360/release"]

all_metrics = {"PSNR": [], "SSIM": [], "LPIPS": []}
print(output_dirs)

for scene in scenes:
    print(scene,end=" ")
    for output in output_dirs:
        json_file = f"{output}/{scene}/results.json"
        data = json.load(open(json_file))
        data = data['ours_30000']
        
        for k in ["PSNR", "SSIM", "LPIPS"]:
            all_metrics[k].append(data[k])
            print(f"{data[k]:.3f}", end=" ")
    print()

latex = []
for k in ["PSNR", "SSIM", "LPIPS"]:
    numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
    print(numbers)
    numbers = [numbers]
    if k == "PSNR":
        numbers = [f"{x:.2f}" for x in numbers]
    else:
        numbers = [f"{x:.3f}" for x in numbers]
    latex.extend(numbers)
print(" & ".join(latex))