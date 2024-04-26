import os
import numpy as np

training_list = [
    'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'
]

scenes = training_list

output_dirs = ["exp_TNT/release"]

all_metrics = {"precision": [], "recall": [], "f-score": []}
print(output_dirs)

for scene in scenes:
    print(scene,end=" ")
    for output in output_dirs:
        # precision
        precision_file = os.path.join(output, scene, f"test/ours_30000/fusion/evaluation/{scene}.precision.txt")
        # precision_file = os.path.join(output, scene, f"test/ours_30000/tsdf/evaluation/{scene}.precision.txt")
        
        precision = np.loadtxt(precision_file)
        precision = precision[precision.shape[0]//5]
        print(precision, end=" ")
        
        # recall
        recall_file = os.path.join(output, scene, f"test/ours_30000/fusion/evaluation/{scene}.recall.txt")
        # recall_file = os.path.join(output, scene, f"test/ours_30000/tsdf/evaluation/{scene}.recall.txt")
        
        recall = np.loadtxt(recall_file)
        recall = recall[recall.shape[0]//5]
        print(recall, end=" ")
        
        # f-score
        f_score = 2 * precision * recall / (precision + recall)
        print(f_score)
        
        all_metrics["precision"].append(precision)
        all_metrics["recall"].append(recall)
        all_metrics["f-score"].append(f_score)


latex = []
for k in ["precision","recall", "f-score"]:
    numbers = all_metrics[k]
    mean = np.mean(numbers)
    numbers = numbers + [mean]
    
    numbers = [f"{x:.2f}" for x in numbers]
    print(k, " & ".join(numbers))
    latex.extend(numbers[-1:])
    
print(" & ".join(latex))