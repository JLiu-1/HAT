import cv2
import numpy as np
import os.path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CesmLrImageDataset(data.Dataset):
    
    def __init__(self, opt):
        super(CesmPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.size_x=opt['size_x'] if 'size_x' in opt else 480
        self.size_y=opt['size_y'] if 'size_y' in opt else 480

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        #img_gt = imfrombytes(img_bytes, float32=True)
        img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))

        # modcrop
        size_h, size_w, _ = img_lq.shape
        #size_h = size_h - size_h % scale
        #size_w = size_w - size_w % scale
        #img_gt = img_gt[0:size_h, 0:size_w, :]

        # generate training pairs
        size_h = max(size_h, self.opt['lq_size'])
        size_w = max(size_w, self.opt['lq_size'])
        #print(img_gt.shape)
        img_lq = cv2.resize(img_gt, (size_w, size_h,1))
        #print(img_gt.shape)
        #img_lq = imresize(img_gt, 1 / scale)

        #img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # augmentation for training
        '''
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
        '''
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        
        

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'lq': img_lq,'lq_path': lq_path}#temp

    def __len__(self):
        return len(self.paths)
