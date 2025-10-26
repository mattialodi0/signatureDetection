import os 
import shutil

source = "datasets/sd02/data"
dest = "datasets/NIST/images"

os.makedirs(dest, exist_ok=True)

for dir1 in os.listdir(source):
    i = 0
    dir_path1 = os.path.join(source, dir1)
    for dir2 in os.listdir(dir_path1):
        dir_path2 = os.path.join(dir_path1, dir2)
        if os.path.isdir(dir_path2):
            for file in os.listdir(dir_path2):
                if file.endswith(".png") and file.startswith(dir2 + "_01"):
                    src_file = os.path.join(dir_path2, file)
                    dst_file = os.path.join(dest, file)
                    shutil.copy(src_file, dst_file)
            i += 1
            if i > 12:
                break