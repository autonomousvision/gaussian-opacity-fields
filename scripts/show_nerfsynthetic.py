import os
import json
import numpy as np
import trimesh

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

output_dirs = ["exp_nerf_synthetic/release"]

results = []
for scene in scenes:
    print(scene,)
    for output in output_dirs:
        json_file = f"{output}/{scene}/results.json"
        data = json.load(open(json_file))
        data = data['ours_30000'] if 'ours_30000' in data else data['ours_7000']

        iteration = "30K iter: "
        point_cloud_file = f"{output}/{scene}/point_cloud/iteration_30000/point_cloud.ply"
        if not os.path.exists(point_cloud_file):
            point_cloud_file = f"{output}/{scene}/point_cloud/iteration_7000/point_cloud.ply"
            iteration = "7K iter: "
        print(iteration, data.values(), trimesh.load(point_cloud_file).vertices.shape)
        results.append(data['PSNR'])

results = np.array(results).reshape(8, -1)

print("===================")
print("PSNR:")
print(results)
print("mean:")
print(results.mean(axis=0))

