
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
class Sci3DDataset(data.Dataset):
    
    def __init__(self, opt):
        super(Sci3DDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        self.size_x=opt['size_x'] if 'size_x' in opt else 64
        self.size_y=opt['size_y'] if 'size_y' in opt else 64
        self.size_z=opt['size_z'] if 'size_z' in opt else 64
        self.max=opt['global_max'] if 'global_max' in opt else None
        self.min=opt['global_min'] if 'global_min' in opt else None
        if 'noise_rate' in opt:
            self.noise_rate=opt['noise_rate'] 
            if 'noise_type' in opt:
                self.noise_type=opt['noise_type']
            else:
                self.noise_type='uniform'
        else:
            self.noise_rate=0
            self.noise_type=None 


      
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,self.size_z,1))
        gmax=np.max(img_gt) if self.max is None else self.max
        gmin=np.min(img_gt) if self.min is None else self.min
        if gmax!=gmin:
            img_gt=(img_gt-gmin)/(gmax-gmin)

        # modcrop

        size_h, size_w,size_d, _ = img_gt.shape
        size_h = size_h - size_h % scale
        size_w = size_w - size_w % scale
        size_d = size_d - size_d % scale
        img_gt = img_gt[0:size_h, 0:size_w, 0:size_d, :]

        # generate training pairs
        #print(img_gt.shape)
        #img_gt = cv2.resize(img_gt, (size_w, size_h,1))
        #print(img_gt.shape)
        #img_lq = imresize(img_gt, 1 / scale)
        img_lq=img_gt[::scale,::scale,::scale,:]

        





        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            lq_size=gt_size //scale
            x_start=np.random.randint(size_h//scale-lq_size)
            y_start=np.random.randint(size_w//scale-lq_size)
            z_start=np.random.randint(size_d//scale-lq_size)


        
            img_gt=img_gt[scale*x_start:scale*x_start+gt_size,scale*y_start:scale*y_start+gt_size,scale*z_start:scale*z_start+gt_size,:]
            img_lq=img_gt[::scale,::scale,::scale,:]
            if self.noise_rate!=0:
                rng=gmax-gmin
                if self.noise_type=='uniform':
                    img_lq+=np.random.uniform(low=-rng*self.noise_rate,high=rng*self.noise_rate,size=img_lq.shape)
                else:
                    img_lq+=np.random.normal(loc=0.0,scale=rng*self.noise_rate,size=img_lq.shape)
            



           
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
        
            img = torch.from_numpy(np.array(img.transpose(3, 0, 1,2)))
            
            img = img.float()
            return img

        img_gt=_totensor(img_gt)
        img_lq=_totensor(img_lq)



        # normalize
        #if self.mean is not None or self.std is not None:
         #   normalize(img_lq, self.mean, self.std, inplace=True)
         #   normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path,'lq_path': gt_path,'max':gmax,'min':gmin}#temp

    def __len__(self):
        return len(self.paths)

