import os 
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


from pathlib import Path
import pickle

import torch
import cv2

from turbojpeg import TurboJPEG

import concurrent.futures
import glob

import imageio


def create_gif_from_numpy_arrays(arrays, output_filename='output.gif', duration=0.1):
    """
    Create a GIF from a list of numpy arrays.

    Parameters:
    -----------
    arrays : list
        List of numpy arrays, each with shape (height, width, 3) representing RGB images
    output_filename : str
        Name of the output GIF file
    duration : float
        Duration of each frame in seconds
    """
    # Check if the arrays have the right format
    if not arrays:
        raise ValueError("Empty list of arrays provided")

    # Convert arrays to uint8 if they're not already
    processed_arrays = []
    for i, img in enumerate(arrays):
        if img.dtype != np.uint8:
            # Normalize if the arrays are in float format
            if img.dtype in [np.float32, np.float64]:
                # Assume values are in [0,1] range
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                # Or values might be in other ranges
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Ensure arrays have the right shape (height, width, 3)
        if img.shape[-1] != 3:
            raise ValueError(f"Array at index {i} doesn't have 3 channels. Shape: {img.shape}")

        processed_arrays.append(img)

    # Write the GIF
    imageio.mimsave(output_filename, processed_arrays, duration=duration)
    print(f"GIF saved as {output_filename}")


class AlohaRightToLeft(tfds.core.GeneratorBasedBuilder):
    """
    Convert a Hugging Face dataset into a TFDS-style episodic dataset with metadata.
    """
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, **kwargs):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset.
            episodes (List[int]): List of episode indices to load. Default is None (load all).
        """
        dataset_name = "aloha_right_to_left_transfer_diverse"
        self.dataset_name = dataset_name
        self.raw_data_path = "/home/hongyi/DATA/aloha_xi_new_dataset/right_to_left_tranfer_single_cube_23_04"
        super().__init__()

    def _info(self):
        """Define dataset information and features."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Converted from the Hugging Face dataset {self.dataset_name}.",
            features=tfds.features.FeaturesDict({
                    'steps': tfds.features.Dataset(
                    {
                        'is_first': tf.bool,
                        'is_last': tf.bool,
                        'observation': tfds.features.FeaturesDict({
                            'state': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                            'images_top': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8), #(340, 420, 3)
                            'images_wrist_left': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                            'images_wrist_right': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                        }),
                        'action': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'timestamp': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'frame_index': tfds.features.Tensor(shape=(), dtype=tf.int32),
                        'is_terminal': tfds.features.Tensor(shape=(), dtype=tf.bool),
                        'language_instruction': tfds.features.Text(),
                        'discount': tfds.features.Tensor(shape=(), dtype=tf.float32),
                    }
                ),
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Tensor(shape=(), dtype=tf.int32),
                })
            }),
        )

    def _split_generators(self, dl_manager):
        """Specify dataset splits."""
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self):
        """Yield examples grouped by episodes."""
        path_list = get_sorted_folders(self.raw_data_path)
        for path in path_list:
            data = process_episode_data(path)
            yield path, data



def process_episode_data(folder_path):
    """
    Reads a .parquet file and converts it into a NumPy array.

    Parameters:
        file_path (str): The path to the .parquet file.

    Returns:
        numpy.ndarray: A NumPy array containing the data from the .parquet file.
    """
    top_cam_path = os.path.join(folder_path, "images/CAM_TOP_orig")
    wrist_left_cam_path = os.path.join(folder_path, "images/CAM_LEFT_orig")
    wrist_right_cam_path = os.path.join(folder_path, "images/CAM_RIGHT_orig")

    leader_joint_path = os.path.join(folder_path, "leader_joint_pos.pt")
    follower_joint_path = os.path.join(folder_path, "follower_joint_pos.pt")

    try:

        leader_joint_pos = torch.load(leader_joint_path)
        follower_joint_pos = torch.load(follower_joint_path)

        top_cam_vector = create_img_vector(top_cam_path, len(leader_joint_pos))
        wrist_left_cam_vector = create_img_vector(wrist_left_cam_path, len(leader_joint_pos))
        wrist_right_cam_vector = create_img_vector(wrist_right_cam_path, len(leader_joint_pos))

        steps = []

        for idx in range(len(leader_joint_pos)-1):
            steps.append({
                'is_first': idx == 0,
                'is_last': idx == len(leader_joint_pos) - 2,
                'observation': {
                    'images_wrist_left': wrist_left_cam_vector[idx],
                    'images_wrist_right': wrist_right_cam_vector[idx],
                    'images_top': top_cam_vector[idx],
                    'state': follower_joint_pos[idx]
                },
                'action': leader_joint_pos[idx+1],
                'reward': 0.0,
                'language_instruction': "Pick up the yellow cube with right arm, transfer it from the right arm to the left arm and then go to a safe position.",
                'is_terminal': idx == len(leader_joint_pos) - 2,
                'discount': 1.0,
                'timestamp': idx,
                'frame_index': idx,
            })
        return {'steps': steps, 'episode_metadata': {'episode_id': 0}}

    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return None


def create_img_vector(img_folder_path, trajectory_length):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.jpg'))
    img_paths =sorted(img_paths,
        key=lambda path: float(os.path.splitext(os.path.basename(path))[0]))
    
    assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        cam_list.append(img_array)

    # create_gif_from_numpy_arrays(cam_list, output_filename='output_flip_2.gif', duration=0.1)

    return cam_list


def get_sorted_folders(base_path):
    base_dir = Path(base_path)
    
    if not base_dir.exists() or not base_dir.is_dir():
        raise ValueError(f"The path {base_path} does not exist or is not a directory")
    
    subfolders = [f for f in base_dir.iterdir() if f.is_dir()]
    
    sorted_subfolders = sorted(subfolders, key=lambda x: x.name)
    
    return [str(folder) for folder in sorted_subfolders]


if __name__ == "__main__":
    folder_path = "/home/hongyi/DATA/aloha_xi_new_dataset/right_to_left_tranfer_single_cube_23_04"

    sorted_paths = get_sorted_folders(folder_path)
    
    for idx in range(len(sorted_paths)):
        step = process_episode_data(sorted_paths[idx+1])
        print(step)

    # Load the dataset
    # ds = tfds.load("real_franka_fold")

    # for episode in ds['train']:
    #     for step in episode['steps']:
    #         print(step)
    #     break