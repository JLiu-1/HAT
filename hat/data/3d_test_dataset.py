
import cv2
import numpy as np
import os.path as osp
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Sci3DTestDataset(data.Dataset):
    
    def __init__(self, opt):
        super(Sci3DTestDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_lq']
        self.size_x=opt['size_x'] if 'size_x' in opt else 64
        self.size_y=opt['size_y'] if 'size_y' in opt else 64
        self.size_z=opt['size_z'] if 'size_z' in opt else 64

      
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,self.size_z,1))
        lmax=np.max(img_lq)
        lmin=np.min(img_lq)
        if lmax!=lmin:
            img_lq=(img_lq-lmin)/(lmax-lmin)
        print(lmin,lmax)
        print(np.min(img_lq),np.max(img_lq))
        # modcrop

        size_h, size_w,size_d, _ = img_lq.shape
    




        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # augmentation for training
    
        # BGR to RGB, HWC to CHW, numpy to tensor


        def _totensor(img):
        
            img = torch.from_numpy(img.transpose(3, 0, 1,2))
            
            img = img.float()
            return img


        img_lq=_totensor(img_lq)



        # normalize
        #if self.mean is not None or self.std is not None:
         #   normalize(img_lq, self.mean, self.std, inplace=True)
         #   normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'lq_path': lq_path,'lmax':lmax,'lmin':lmin}#temp

    def __len__(self):
        return len(self.paths)

