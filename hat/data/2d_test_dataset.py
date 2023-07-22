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
class Sci2DTestDataset(data.Dataset):
    
    def __init__(self, opt):
        super(Sci2DTestDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.size_x=opt['size_x'] if 'size_x' in opt else 480
        self.size_y=opt['size_y'] if 'size_y' in opt else 480
        self.max=opt['global_max'] if 'global_max' in opt else None
        self.min=opt['global_min'] if 'global_min' in opt else None

        self.slices_from_3d=False
        self.size_z=None
        if 'size_z' in opt:
            self.slices_from_3d=True;
            self.size_z=opt['size_z'] 

       
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
        if not self.slices_from_3d:
            img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))
        else:
            filename=osp.basename(lq_path)
            if filename[0]=='y':
                img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_z,1))
            elif filename[0]=='x':
                img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_y,self.size_z,1))
            else:
                img_lq=np.frombuffer(img_bytes,dtype=np.float32).reshape((self.size_x,self.size_y,1))
        lmax=np.max(img_lq) if self.max is None else self.max
        lmin=np.min(img_lq) if self.min is None else self.min
        if lmax!=lmin:
            img_lq=(img_lq-lmin)/(lmax-lmin)

        # modcrop
        size_h, size_w, _ = img_lq.shape

        
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # augmentation for training
        
        img_lq = img2tensor(np.array(img_lq), bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'lq': img_lq,'lq_path': lq_path,'max':lmax,'min':lmin}#temp

    def __len__(self):
        return len(self.paths)
