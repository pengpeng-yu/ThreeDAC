import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


import argparse
import sys
import os
from gpcc_test import gpcc_test
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

os.chdir("..") 


import numpy as np

from Haar3D_info import haar3D, inv_haar3D
import torch
import torch.optim as optim


from dataloaders.scannet_pcc import ScannetPCC
import time

import utils
from InitialCoding import InitialCoding_ScanNet as InitialCoding
from InitialCoding import Construct_InitialCoding_Info_ScanNet as Construct_InitialCoding_Info
from InitialCoding import Construct_InterChannel_Info_ScanNet as Construct_InterChannel_Info



parser = argparse.ArgumentParser()
parser.add_argument('--Qstep', type=int, default=10)
parser.add_argument('--dir_path', type=str, default='dataset/Scannet_v2/')
parser.add_argument('--depth', type=int, default=9)
opt = parser.parse_args()
print(opt)





dir_path=opt.dir_path
out_model_dir=os.path.join(r'checkpoints/', BASE_DIR.split('/')[-1])

if not os.path.exists(out_model_dir):
    os.makedirs(out_model_dir)



depth = opt.depth
Qstep = opt.Qstep











train_dataset = ScannetPCC(dir_path=dir_path, mode='train', depth=depth)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0) 

test_dataset = ScannetPCC(dir_path=dir_path, mode='val', depth=depth)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)        



# train_dataset.data_path_list=train_dataset.data_path_list[::10]
# test_dataset.data_path_list=test_dataset.data_path_list[::10]



from models.EntropyBottleneck import EntropyBottleneck


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entropy_bottleneck = EntropyBottleneck(channels=3, content_chann=24).cuda()





from models.InitialCodingModule import InitialCodingModule
from models.InterChannelModule import InterChannelModule

initial_coding_module = InitialCodingModule(in_channel=11, out_channel=8, depth=depth).cuda()
inter_channel_module = InterChannelModule(out_channel=4, depth=depth).cuda()






model_path=out_model_dir+r'/eb_q%d_best.pth'%(Qstep)
initial_coding_module_path=out_model_dir+r'/initial_coding_module_q%d_best.pth'%(Qstep) 
inter_channel_module_path=out_model_dir+r'/inter_channel_module_q%d_best.pth'%(Qstep)


entropy_bottleneck.load_state_dict(torch.load(model_path))
initial_coding_module.load_state_dict(torch.load(initial_coding_module_path))
inter_channel_module.load_state_dict(torch.load(inter_channel_module_path))







with torch.no_grad():
    test_loss_sum = 0
    psnr_list = []
    entropy_bottleneck.eval() 
    initial_coding_module.eval() 
    inter_channel_module.eval()
    bpp_list_gpcc = []
    psnr_list_gpcc = []
 
    
    training=False       
    
    for batch_id, data in enumerate(test_dataloader):
        print(batch_id)
        
        
        # load point cloud
        points, colors = data
        points, colors = points[0].numpy(), colors[0].numpy()
        colors = colors*255
        bpp_gpcc, psnr_gpcc = gpcc_test(points, colors)
        bpp_list_gpcc.append(bpp_gpcc)
        psnr_list_gpcc.append(psnr_gpcc)
        colors = utils.RGB2YUV(colors)
        
        
            
        # initial coding
        res = InitialCoding(points, colors, depth, Qstep)
        
        # side information
        CT = res['CT']
        depth_CT, w, node_xyz = res['depth_CT'], res['w'], res['node_xyz']
        low_freq, cyuv, crgb = res['low_freq'], res['cyuv'], res['crgb']
        iCT_low, iW, iPos = res['iCT_low'], res['iW'], res['iPos']
        iyuv, irgb = res['iyuv'], res['irgb']
        
        points, colors = torch.tensor(points), torch.tensor(colors)
        
        
        
        # time_preporocess1_end = time.time()
        # print('time_preporocess1_end', time_preporocess1_end-time_preporocess1)
        
  
     
        # dpeth level of context tensor
        level_list = res['level_list']
            
            
        time_preporocess1 = time.time()           
            
        
    
        # convert side information to context tensor
        res_ini_info = Construct_InitialCoding_Info(res, Qstep, depth)
        high_freq_info = res_ini_info['high_freq_info']
        high_freq_nodes_sp = res_ini_info['high_freq_nodes_sp']
        low_freq_nodes_sp = res_ini_info['low_freq_nodes_sp']    
        
        
        res_inter = Construct_InterChannel_Info(res, Qstep, depth)
        inter_channel_info_y = res_inter['inter_channel_info_y']
        inter_channel_info_yu = res_inter['inter_channel_info_yu']
        
        high_freq_nodes_sp = res_inter['high_freq_nodes_sp']    
        high_freq_nodes_y_sp = res_inter['high_freq_nodes_y_sp']    
        high_freq_nodes_yu_sp = res_inter['high_freq_nodes_yu_sp']            
        
        
        
        
        
        
        
        
        
        
        time_preporocess_end = time.time()
        #print('time_preporocess_end', time_preporocess_end-time_preporocess)
        
      
        time_spvcnn = time.time()     
        
        
        low_freq_info = initial_coding_module.spatial_aggregate(low_freq_nodes_sp.cuda(), 
                                                                high_freq_nodes_sp.cuda()
                                                                )
        
        
        
        spatial_info_y, spatial_info_yu = inter_channel_module.spatial_aggregate(high_freq_nodes_y_sp.cuda(),
                                                                                 high_freq_nodes_yu_sp.cuda(),
                                                                                 high_freq_nodes_sp.cuda()
                                                                                 )
        
        
        
        
        
        time_spvcnn_end = time.time()
        #print('time_spvcnn_end', time_spvcnn_end-time_spvcnn)
        
        
        
        
        DC_coef = CT[0:1]/Qstep
        # del DC coeff
        CT = CT[1:]
        w = w[1:]
        depth_CT = depth_CT[1:]
        node_xyz = node_xyz[1:]
        low_freq = low_freq[1:]        
        
        cyuv = cyuv[1:]
        crgb = crgb[1:]        
        
        CT_q = CT/Qstep
        
        
        
        
        
        time_lk = time.time()
        
        
        bpv_sum = 0
        for i in range(depth*3):
            
            
            mask = depth_CT==i
            if torch.sum(mask)<=1:
                continue
         
            
            
            context_initial_coding = initial_coding_module(high_freq_info[level_list[i]:level_list[i+1]].cuda(), 
                                                         low_freq_info[level_list[i]:level_list[i+1]], 
                                                         depth_idx=i)
            context_inter_channel = inter_channel_module(inter_channel_info_y[level_list[i]:level_list[i+1]].cuda(), 
                                                         inter_channel_info_yu[level_list[i]:level_list[i+1]].cuda(), 
                                                         spatial_info_y[level_list[i]:level_list[i+1]], 
                                                         spatial_info_yu[level_list[i]:level_list[i+1]], 
                                                         depth_idx=i)
            
            
            
            
            # fusion module is merged to the entropy bottleneck
            context = torch.cat((context_initial_coding[:,None].repeat(1,context_inter_channel.shape[1],1), 
                                 context_inter_channel), -1)
            
            
            
            
            
            CT_q_tilde, likelihood = entropy_bottleneck(torch.tensor(CT_q[mask]).float().cuda(), context, training, device)
            bpv = torch.sum(torch.log(likelihood)) / -(torch.log(torch.Tensor([2.0]).to(device)))
            
            bpv_sum += bpv
            
            
        time_lk_end = time.time()
        #print('time_lk_end', time_lk_end-time_lk)     
        
        
        
        
        
        bpv_sum = bpv_sum / CT_q.shape[0]
            
                    
   
        test_loss = bpv_sum
        test_loss_sum += test_loss.item()
        
        psnr = utils.eval_rec(points.numpy(), colors.numpy(),
                              torch.cat((DC_coef, CT_q), 0).numpy(), 
                              Qstep, depth, inv_haar3D) 
        psnr_list.append(psnr)            
            
        # print(batch_id, train_loss.item())        
        # if batch_id==100:
        #     break
    
        # break
    
    print('Average test loss and psnr: %f, %f' %
          ((test_loss_sum / len(test_dataloader)), np.mean(psnr_list))
          )
    print('GPCC bpp:', np.mean(bpp_list_gpcc))
    print('GPCC psnr:', np.mean(psnr_list_gpcc))











