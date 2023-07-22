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
class Sci2DDataset(data.Dataset):
    
    def __init__(self, opt):
        super(Sci2DDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        self.size_x=opt['size_x'] if 'size_x' in opt else 480
        self.size_y=opt['size_y'] if 'size_y' in opt else 480
        self.max=opt['global_max'] if 'global_max' in opt else None
        self.min=opt['global_min'] if 'global_min' in opt else None
        self.use_hflip=opt['use_hflip'] if 'use_hflip' in opt else False
        self.use_rot=opt['use_rot'] if 'use_rot' in opt else False

        self.slices_from_3d=False
        self.size_z=None
        if 'size_z' in opt:
            self.slices_from_3d=True;
            self.size_z=opt['size_z'] 


        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        #img_gt = imfrombytes(img_bytes, float32=True)
        if not self.slices_from_3d:
            img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))
        else:
            filename=osp.basename(gt_path)
            if filename[0]=='y':
                img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_z,1))
            elif filename[0]=='x':
                img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_y,self.size_z,1))
            else:
                img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))
        gmax=np.max(img_gt) if self.max is None else self.max
        gmin=np.min(img_gt) if self.min is None else self.min
        if gmax!=gmin:
            img_gt=(img_gt-gmin)/(gmax-gmin)

        # modcrop
        size_h, size_w, _ = img_gt.shape
        size_h = size_h - size_h % scale
        size_w = size_w - size_w % scale
        img_gt = img_gt[0:size_h, 0:size_w, :]

        # generate training pairs
        size_h = max(size_h, self.opt['gt_size'])
        size_w = max(size_w, self.opt['gt_size'])
        #print(img_gt.shape)
        #img_gt = cv2.resize(img_gt, (size_w, size_h,1))
        #print(img_gt.shape)
        #img_lq = imresize(img_gt, 1 / scale)
        img_lq=img_gt[::scale,::scale,:]

        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)#to customize

            #print(img_gt.shape)
            #print(img_lq.shape)
            #flip, rotation
            if self.use_hflip or self.use_rot:
                img_gt, img_lq = augment([img_gt, img_lq], self.use_hflip, self.use_rot)#to customize
        '''
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
        '''
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([np.array(img_gt), np.array(img_lq)], bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path,'lq_path': gt_path,'max':gmax,'min':gmin}#temp

    def __len__(self):
        return len(self.paths)
