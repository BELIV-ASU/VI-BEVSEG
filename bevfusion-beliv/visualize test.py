# import pickle
# import matplotlib.pyplot as plt
#
# # Step 1: Load the .pkl file
# with open('nuscenes/nuscenes_infos_train.pkl', 'rb') as file:
#     data = pickle.load(file)
#
# # Step 2: Print the type of data (optional)
# print(type(data))
#
# # Step 3: Visualize the data (assuming it's a simple list or DataFrame)
# if isinstance(data, list):
#     plt.plot(data)  # Simple line plot if it's a list of numerical values
#     plt.show()
# elif isinstance(data, dict):
#     # Handle dictionary appropriately depending on its structure
#     pass
# elif "DataFrame" in str(type(data)):
#     data.plot()  # Use DataFrame's built-in plotting method
#     plt.show()


# import json
# with open('/media/jingxiong/HaoranSSD/V2X-SIM/v1.0-mini/sample_annotation.json', 'r') as file:
#     data = json.load(file)
#
# for i in range(len(data)):
#     data[i]['num_lidar_pts'] = max(data[i]['num_lidar_pts'])
#     data[i]['num_radar_pts'] = 0
#
# with open('/media/jingxiong/HaoranSSD/V2X-SIM/v1.0-mini/filter_sample_annotation.json', 'w', newline='\n') as file:
#     json.dump(data, file, indent=4)
# print(data)


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the .npz file
# npz_file = np.load('/media/jingxiong/HaoranSSD/V2X-SIM/samples/BEV_TOP_id_0/scene_1_000006.npz')
#
# # Print the contents
# print("Contents of the .npz file:")
# for array_name in npz_file.files:
#     print(f"{array_name}: shape = {npz_file[array_name].shape}")
#
# # Function to visualize an array
# def visualize_array(array, title="Array"):
#     plt.figure()
#     plt.imshow(array, cmap='gray')
#     plt.title(title)
#     plt.colorbar()
#     plt.show()
#
# # Visualize each array
# for array_name in npz_file.files:
#     array = npz_file[array_name]
#     if array.ndim == 2:  # Visualize 2D arrays directly
#         visualize_array(array, title=array_name)
#     elif array.ndim == 3 and array.shape[0] <= 3:  # Visualize 3D arrays as RGB if possible
#         if array.shape[0] == 1:  # If single channel, visualize as grayscale
#             visualize_array(array[0], title=array_name)
#         else:  # If multiple channels, visualize as RGB
#             plt.figure()
#             plt.imshow(array.transpose(1, 2, 0))
#             plt.title(array_name)
#             plt.show()
#     else:
#         print(f"Skipping visualization of {array_name}: unsupported shape {array.shape}")
#
# # Close the .npz file
# npz_file.close()


# visualize the pcd file
import open3d as o3d
import numpy as np

# Function to read binary PCD file
def read_pcd_bin(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 5))
    return point_cloud[:, :3]  # We only need the x, y, z coordinates

# Read the PCD file
file_path = '/media/jingxiong/HaoranSSD/V2X-SIM/lidar_concat/sweeps/LIDAR_TOP_id_1/scene_1_000046.pcd.bin'
points = read_pcd_bin(file_path)

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])