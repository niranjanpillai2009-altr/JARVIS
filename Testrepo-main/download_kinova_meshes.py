"""
Download Kinova Gen 3 mesh files from the official Kinova ROS repository
"""
import os
import urllib.request
import zipfile
import shutil

def download_kinova_meshes():
    """Download and extract Kinova Gen 3 mesh files"""
    
    # Create meshes directory
    mesh_dir = "kinova_meshes"
    os.makedirs(mesh_dir, exist_ok=True)
    
    print("Downloading Kinova Gen 3 mesh files from official repository...")
    
    # GitHub raw content URLs for Kinova Gen 3 meshes (7DOF version)
    mesh_files = [
        ("base_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/base_link.STL"),
        ("shoulder_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/shoulder_link.STL"),
        ("half_arm_1_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/half_arm_1_link.STL"),
        ("half_arm_2_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/half_arm_2_link.STL"),
        ("forearm_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/forearm_link.STL"),
        ("spherical_wrist_1_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/spherical_wrist_1_link.STL"),
        ("spherical_wrist_2_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/spherical_wrist_2_link.STL"),
        ("bracelet_with_vision_link.STL", "https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/noetic-devel/kortex_description/arms/gen3/7dof/meshes/bracelet_with_vision_link.STL"),
    ]
    
    success_count = 0
    for filename, url in mesh_files:
        try:
            filepath = os.path.join(mesh_dir, filename)
            print(f"Downloading {filename}...", end=" ")
            urllib.request.urlretrieve(url, filepath)
            print("✓")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\nDownloaded {success_count}/{len(mesh_files)} mesh files")
    print(f"Meshes saved to: {os.path.abspath(mesh_dir)}")
    
    return mesh_dir

if __name__ == "__main__":
    mesh_dir = download_kinova_meshes()
