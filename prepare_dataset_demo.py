import nibabel as nib
import numpy as np
import cv2
import os
from tqdm import tqdm

save_dir = ["/path/tp/save/CT", "/path/tp/save/PET"]
read_dir = "/path/to/read"

fpy = os.listdir(read_dir)
rpy = os.listdir(save_dir[0])
charf = ["CT.nii.gz", "PET.nii.gz"]

for dir_name in tqdm(fpy):
    sdirpaths = []
    for i in range(len(save_dir)):
        if dir_name in rpy:
            continue
        dirpath = os.path.join(read_dir, dir_name)
        sdirpath = os.path.join(save_dir[i], dir_name)
        qpy = os.listdir(dirpath)
        flag = False
        for name in charf:
            if name not in qpy:
                flag = True
                break
        os.mkdir(sdirpath)
        sdirpaths.append(sdirpath)
    if flag:
            continue
    flag = True
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    for ind, niiname in enumerate(charf):
        try:
            nii_file = os.path.join(dirpath, niiname)
            nii_image = nib.load(nii_file)
            nii_data = nii_image.get_fdata()
            tfile_name = niiname.replace(".nii.gz", "")
        except:
            continue

        for layer in range(0, nii_data.shape[-1]):
            if not flag:
                if layer >= len(xmin):
                    break
            try:
                niis = nii_data[:,:,layer:layer+1]
                if niis.max() == niis.min():
                    continue
                niist = niis.reshape(-1)
                niist.sort()
                niilow = niist[niist.shape[0] * 1 // 10]
                niihigh = niist[niist.shape[0] * 99 // 100]
                niis[niis<niilow] = niilow
                niis[niis>niihigh] = niihigh
                niis = (niis - niis.min()) / (niis.max() - niis.min())
                if flag:
                    x = np.linspace(1, niis.shape[1], niis.shape[1])
                    y = np.linspace(1, niis.shape[0], niis.shape[0])
                    X, Y = np.meshgrid(x, y)
                    npp = niis.copy()
                    niism = (npp>0.01)
                    x0 = X * niism[:,:,0]
                    y0 = Y * niism[:,:,0]
                    x0 = x0[x0>0]
                    y0 = y0[y0>0]
                    xmin.append(int(x0.min()))
                    xmax.append(int(x0.max()))
                    ymin.append(int(y0.min()))
                    ymax.append(int(y0.max()))
                niis = niis[ymin[layer]:ymax[layer], xmin[layer]:xmax[layer]]
                np.save(os.path.join(sdirpaths[ind], "{}_{}.npy".format(tfile_name, layer)), niis)
            except:
                continue
        flag = False
        del nii_image
        del nii_data

