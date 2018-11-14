from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm_notebook
import numpy as np
import os


def get_img_time(img_name):
    time_parts = img_name[2:].split('-')
    return float(time_parts[0])


def get_synched_frames(images_dir):
    sync_frames = []
    files = list(os.walk(images_dir))[0][-1]
    depth_files = [x for x in files if x[0] == 'd']
    rgb_files = [x for x in files if x[0] == 'r']
    depth_files.sort()
    rgb_files.sort()
    j = 0
    for i in range(len(depth_files)):
        depth_time = get_img_time(depth_files[i])
        while j + 1 < len(rgb_files):
            cur_rgb_time = get_img_time(rgb_files[j])
            next_rgb_time = get_img_time(rgb_files[j + 1])
            cur_tdiff = abs(cur_rgb_time - depth_time)
            next_tdiff = abs(next_rgb_time - depth_time)
            if next_tdiff < cur_tdiff:
                j += 1
            else:
                break
        sync_frames.append((depth_files[i], rgb_files[j]))
    return sync_frames


def extract_images(images_dir, filename_matches, data_percent=5.):
    depths = []
    rgbs = []
    np.random.shuffle(filename_matches)
    filename_matches = filename_matches[:int(len(filename_matches) * data_percent / 100.)]
    for line in filename_matches:
        depth_filename, rgb_filename = line
        try:
            rgb_image = imread(os.path.join(images_dir, rgb_filename))
            depth_image = imread(os.path.join(images_dir, depth_filename))
            depths.append(depth_image)
            rgbs.append(rgb_image)
        except:
            continue
    return depths, rgbs


def process_images(depths, rgbs):
    assert(len(rgbs) == len(depths))
    # create dir 'tmp' to save images into, during the work
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
        
    processedDepths = []
    processedRGBs = []
    for i in tqdm_notebook(np.arange(len(depths))):
        # save images to process them using matlab scripts
        depth_filename = "../tmp/depth_" + str(i) + ".png"
        rgb_filename = "../tmp/rgb_" + str(i) + ".png"
        imsave(depth_filename, depths[i])
        imsave(rgb_filename, rgbs[i])
        
        # make depth map projection
        run_script_cmd = "octave ./project_depth_map.m"
        os.system(' '.join([run_script_cmd, depth_filename, rgb_filename, depth_filename, rgb_filename]))
        
        # apply cross-bilateral filter to depth map
        run_script_cmd = "octave ./fill_depth_cross_bf.m"
        os.system(' '.join([run_script_cmd, depth_filename, rgb_filename, depth_filename]))
        
        # finish image process: resize and transpose axes
        depth_img = imread(depth_filename)
        rgb_img = imread(rgb_filename)
        #depth_img = resize(depth_img, (240, 320))
        #rgb_img = resize(rgb_img, (240, 320, 3))
        rgb_img = np.transpose(rgb_img, [2, 0, 1])
        processedDepths.append(depth_img)
        processedRGBs.append(rgb_img)
    
    # delete dir 'tmp'
    # os.system("rm -rf ../tmp")
    return processedDepths, processedRGBs