import sys

import matplotlib

matplotlib.use("TkAgg")  # Use an interactive backend
import h5py
import os
import matplotlib.pyplot as plt
import shutil

key_to_idx = {
    'box': 0,
    'confmaps': 4,
    'cropZone': 0,
    'joints': 3,
    'labeledIdx': 1,
    'points_3d': 1,
}

# path = "/cs/labs/tsevi/lior.kotlar/pose-estimation/training_datasets/random_trainset_201_frames_18_joints.h5"



def readfile(path):
    with (h5py.File(path, "r") as f):
        keys = list(f.keys())
        for key in keys:
            value = f[key]
            print(f'key:{key}, value:{value}')


def cropfile(path):
    directory = os.path.dirname(path)
    file_name_without_ext = os.path.splitext(os.path.basename(path))[0]
    new_file_name = os.path.join(directory, f'{file_name_without_ext}-cropped.h5')
    print(new_file_name)
    shutil.copy(path, new_file_name)
    with h5py.File(new_file_name, "r+") as f:
        keys = list(f.keys())
        for key in keys:
            value = f[key]
            print(f'key:{key}, value:{value}')
            if key == 'box':
                print(f'key:{key}, shape:{value.shape}')
                newbox = value[:10, :, :, :, :]
                del f[key]
                f.create_dataset('box', data=newbox)
                print(newbox.shape)
                continue
            if key == 'confmaps':
                print(f'key:{key}, shape:{value.shape}')
                newconfmaps = value[:, :, :, :, :10]
                del f[key]
                f.create_dataset('confmaps', data=newconfmaps)
                print(newconfmaps.shape)
                continue
            if key == 'cropZone':
                print(f'key:{key}, shape:{value.shape}')
                newcropZone = value[:10, :, :]
                del f[key]
                f.create_dataset('cropZone', data=newcropZone)
                print(newcropZone.shape)
                continue
            if key == 'joints':
                print(f'key:{key}, shape:{value.shape}')
                newjoints = value[:, :, :, :10]
                del f[key]
                f.create_dataset('joints', data=newjoints)
                print(newjoints.shape)
                continue
            if key == 'labeledIdx':
                print(f'key:{key}, shape:{value.shape}')
                newlabeledIdx = value[:, :10]
                del f[key]
                f.create_dataset('labeledIdx', data=newlabeledIdx)
                print(newlabeledIdx.shape)
                continue
            if key == 'points_3D':
                print(f'key:{key}, shape:{value.shape}')
                newpoints_3d = value[:, :10, :]
                del f[key]
                f.create_dataset('points_3D', data=newpoints_3d)
                print(newpoints_3d.shape)
                continue



def cropmovie(path, cropped_movie_length_in_frames):
    newfoldername = 'cropped'
    directory = os.path.dirname(path)
    newfolderpath = os.path.join(directory, newfoldername)
    os.makedirs(newfolderpath, exist_ok=True)
    file_name_without_ext = os.path.splitext(os.path.basename(path))[0]
    new_file_name = os.path.join(newfolderpath, f'{file_name_without_ext}-cropped({str(cropped_movie_length_in_frames)}).h5')
    print(new_file_name)
    shutil.copy(path, new_file_name)
    with h5py.File(new_file_name, 'r+') as f:
        keys = list(f.keys())
        f.items()
        for key in keys:
            if key == 'best_frames_mov_idx':
                newbestframesmovidx = f[key][:, :cropped_movie_length_in_frames]
                del f[key]
                f.create_dataset(key, data=newbestframesmovidx)
                print(newbestframesmovidx.shape)
                continue
            else:
                newvalue = f[key][:cropped_movie_length_in_frames, ...]
                del f[key]
                f.create_dataset(key, data=newvalue)


def main():
    if len(sys.argv) < 3:
        print("Usage: python h5files.py <action> <file_path> [<cropped_movie_length_in_frames> - for 'cm' action]")
        print("Actions: 'r' - read file, 'c' - crop file, 'cm' - crop movie")
        exit(1)
    path = sys.argv[2]
    if not os.path.exists(path):
        print("file doesen't exist")
        exit(1)
    action = sys.argv[1]
    if action == 'r':
        readfile(path)
    elif action == 'c':
        cropfile(path)
    elif action == 'cm':
        if len(sys.argv) < 4:
            print("Please provide the number of frames to crop.")
            exit(1)
        else:
            try:
                cropped_movie_length_in_frames = int(sys.argv[3])
            except ValueError:
                print("Invalid number of frames provided.")
                exit(1)
            if cropped_movie_length_in_frames <= 0:
                print("Number of frames must be a positive integer.")
                exit(1)
        cropmovie(path, cropped_movie_length_in_frames)


if __name__ == '__main__':
    main()
