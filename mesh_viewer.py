import open3d as o3d
import sys
import os
import numpy as np

def load_and_show_ply2(filepath):
    # Load the PLY file
    #mesh = o3d.io.read_point_cloud(filepath)

    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    """
    try:
        # Try loading as a point cloud first
        mesh = o3d.io.read_point_cloud(filepath)
    except:
        # If that fails, try loading as a triangle mesh
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()
    """
    # Visualize the point cloud
    o3d.visualization.draw_geometries([mesh])

def is_triangle_mesh(filepath):
    if filepath.endswith(".obj"):
        return True

    # Check the contents of the PLY file for the 'element face' line
    try:
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                if 'element face' in line:
                    return True
    except Exception as e:
        print(f"Error reading file: {e}")
    return False


def load_and_show_ply(filepath):
    if is_triangle_mesh(filepath):
        mesh = o3d.io.read_triangle_mesh(filepath)
        if False:
            triangles = np.asarray(mesh.triangles)
            # flip triangles
            triangles = triangles[:, [0, 2, 1]]
            mesh = o3d.geometry.TriangleMesh(vertices=mesh.vertices, triangles=o3d.utility.Vector3iVector(triangles))

        mesh.compute_vertex_normals()
        print("read triangle")
    else:
        mesh = o3d.io.read_point_cloud(filepath)
        print("read point cloud")
    # Visualize the geometry (whether it's a point cloud or triangle mesh)
    # o3d.visualization.draw_geometries([mesh])

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()

    # Create a window with the filename as the title
    filename = os.path.basename(filepath)
    # vis.create_window(window_name=filepath)
    vis.create_window(window_name=filepath, width=1600, height=1200)

    # Add the geometry to the visualizer
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.001)
    ctr.set_constant_z_far(1000)
        
    # Run the visualizer
    vis.run()

    # Close the visualizer window
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py path_to_ply_file")
        exit()
    
    filepath = sys.argv[1]
    load_and_show_ply(filepath)


