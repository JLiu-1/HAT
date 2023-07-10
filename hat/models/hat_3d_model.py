import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp
import os
import numpy as np
@MODEL_REGISTRY.register()
class HATModel_3D(SRModel):

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w , self.mod_pad_d = 0, 0, 0
        _, _, h, w,d = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        if d % window_size != 0:
            self.mod_pad_d = window_size - d % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h, 0, self.mod_pad_d), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width, depth = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_depth = depth * self.scale
        output_shape = (batch, channel, output_height, output_width, output_depth)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil( depth/ self.opt['tile']['tile_size'])
        tiles_y = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_z = math.ceil(height/ self.opt['tile']['tile_size'])

        # loop over all tiles
        for z in range(tiles_z):
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # extract tile from input image
                    ofs_x = x * self.opt['tile']['tile_size']
                    ofs_y = y * self.opt['tile']['tile_size']
                    ofs_z = z * self.opt['tile']['tile_size']
                    # input tile area on total image
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + self.opt['tile']['tile_size'], depth )
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + self.opt['tile']['tile_size'],width)
                    input_start_z = ofs_z
                    input_end_z = min(ofs_z + self.opt['tile']['tile_size'],height)

                    # input tile area on total image with padding
                    input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                    input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'],depth )
                    input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                    input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], width)
                    input_start_z_pad = max(input_start_z - self.opt['tile']['tile_pad'], 0)
                    input_end_z_pad = min(input_end_z + self.opt['tile']['tile_pad'],height )

                    # input tile dimensions
                    input_tile_width = input_end_x - input_start_x
                    input_tile_height = input_end_y - input_start_y
                    input_tile_depth = input_end_z - input_start_z
                    tile_idx = z*tiles_y*tiles_x+ y * tiles_x + x + 1
                    input_tile = self.img[:, :,input_start_z_pad:input_end_z_pad, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                    # upscale tile
                    try:
                        if hasattr(self, 'net_g_ema'):
                            self.net_g_ema.eval()
                            with torch.no_grad():
                                output_tile = self.net_g_ema(input_tile)
                        else:
                            self.net_g.eval()
                            with torch.no_grad():
                                output_tile = self.net_g(input_tile)
                    except RuntimeError as error:
                        print('Error', error)
                    print(f'\tTile {tile_idx}/{tiles_x * tiles_y* tiles_z}')

                    # output tile area on total image
                    output_start_x = input_start_x * self.opt['scale']
                    output_end_x = input_end_x * self.opt['scale']
                    output_start_y = input_start_y * self.opt['scale']
                    output_end_y = input_end_y * self.opt['scale']
                    output_start_z = input_start_z * self.opt['scale']
                    output_end_z = input_end_z * self.opt['scale']

                    # output tile area without padding
                    output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                    output_end_x_tile = output_start_x_tile + input_tile_depth * self.opt['scale']
                    output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                    output_end_y_tile = output_start_y_tile + input_tile_width * self.opt['scale']
                    output_start_z_tile = (input_start_z - input_start_z_pad) * self.opt['scale']
                    output_end_z_tile = output_start_z_tile + input_tile_height * self.opt['scale']

                    # put tile into output image
                    self.output[:, :, output_start_z:output_end_z, output_start_y:output_end_y,
                                output_start_x:output_end_x] = output_tile[:, :,output_start_z_tile:output_end_z_tile, output_start_y_tile:output_end_y_tile,
                                                                           output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w, d = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale, 0:d - self.mod_pad_d * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        #print("nidielaile")
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        if save_img:
            if self.opt['is_train']:
                save_img_dir = osp.join(self.opt['path']['visualization'], img_name)
            else:
                
                save_img_dir = osp.join(self.opt['path']['visualization'], dataset_name)
            #print(save_img_dir)
            if not osp.exists(save_img_dir):
                os.makedirs(save_img_dir)
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            lmax=val_data['lmax'][0].item() if 'lmax' in val_data else 1
            lmin=val_data['lmin'][0].item() if 'lmin' in val_data else 0
            #print("val",lmin,lmax)
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            def _tonumpy(tensor,minmax=(0,1)):


                
                print(tensor.shape)
                img_np = tensor[0].numpy()
                img_np = img_np.transpose(1, 2,3, 0)
                img_np = np.squeeze(img_np, axis=3)

                img_np = img_np.astype(np.float32)
                img_np=(img_np-minmax[0])/(minmax[1]-minmax[0])
                print(np.min(img_np),np.max(img_np))
                return img_np


            sr_img = _tonumpy([visuals['result']][0],(lmin,lmax))
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = _tonumpy([visuals['gt']][0],(lmin,lmax))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
               
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.dat')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.dat')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.dat')
                #imwrite(sr_img, save_img_path)
                sr_img.tofile(save_img_path)
                #print("woshinibaba")

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
