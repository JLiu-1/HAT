
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
class 3DDataset(data.Dataset):
    
    def __init__(self, opt):
        super(3DDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        #self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        #self.gt_folder = opt['dataroot_gt']
        self.size_x=opt['size_x'] if 'size_x' in opt else 32
        self.size_y=opt['size_y'] if 'size_y' in opt else 32
        self.size_z=opt['size_z'] if 'size_z' in opt else 32

      
        #self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        #if self.file_client is None:
        #    self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        #gt_path = self.paths[index]
        gt_path=" "
        #img_bytes = self.file_client.get(gt_path, 'gt')
        #img_gt = imfrombytes(img_bytes, float32=True)
        #img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))

        # modcrop
        img_gt=np.random.rand(self.size_x,self.size_y,self.size_z,1).astype(np.float)

        size_h, size_w,size_d, _ = img_gt.shape
        size_h = size_h - size_h % scale
        size_w = size_w - size_w % scale
        size_d = size_d - size_d % scale
        img_gt = img_gt[0:size_h, 0:size_w, 0:size_d, :]

        # generate training pairs
        size_h = max(size_h, self.opt['gt_size'])
        size_w = max(size_w, self.opt['gt_size'])
        size_d = max(size_d, self.opt['gt_size'])
        #print(img_gt.shape)
        #img_gt = cv2.resize(img_gt, (size_w, size_h,1))
        #print(img_gt.shape)
        #img_lq = imresize(img_gt, 1 / scale)
        img_lq=img_gt[::scale,::scale,::scale,:]

        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            #img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)#to customize

            #print(img_gt.shape)
            #print(img_lq.shape)
            #flip, rotation
            #img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])#to customize
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


        def _totensor(img):
        
            img = torch.from_numpy(img.transpose(3, 0, 1,2))
            
            img = img.float()
            return img

        img_gt=_totensor(img_gt)
        img_lq=_totensor(img_lq)



        # normalize
        #if self.mean is not None or self.std is not None:
         #   normalize(img_lq, self.mean, self.std, inplace=True)
         #   normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path,'lq_path': gt_path}#temp

    def __len__(self):
        #return len(self.paths)
        return 100
