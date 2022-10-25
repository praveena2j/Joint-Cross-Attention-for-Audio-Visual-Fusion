from __future__ import absolute_import
from __future__ import division

from torch.nn import init
import torch
import math
from torch import nn
from torch.nn import functional as F
import sys
from .av_crossatten import DCNLayer
from .layer import LSTM

from .audguide_att import BottomUpExtract

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        #self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))


        #self.coattn = DCNLayer(opt.video_size, opt.audio_size, opt.num_seq, opt.dropout)
        self.coattn = DCNLayer(512, 512, 1, 0.6)

        #self.encoder1 = nn.Linear(512, 256)
        #self.encoder2 = nn.Linear(512, 256)
        self.video_attn = BottomUpExtract(512, 512)
        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

     

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        self.init_weights()

    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        #f1 = f1.squeeze(1)
        #f2 = f2.squeeze(1)

        #f1_norm = F.normalize(f1_norm, p=2, dim=2, eps=1e-12)
        #f2_norm = F.normalize(f2_norm, p=2, dim=2, eps=1e-12)

        #fin_audio_features = []
        #fin_visual_features = []
        #vsequence_outs = []
        #asequence_outs = []

        #for i in range(f1_norm.shape[0]):
        #aud_fts = f1_norm[i,:,:]#.transpose(0,1)
        #vis_fts = f2_norm[i,:,:]#.transpose(0,1)

        #audfts = self.encoder1(aud_fts)
        #visfts = self.encoder2(vis_fts)

        video = F.normalize(f2_norm, dim=-1)
        audio = F.normalize(f1_norm, dim=-1)
      
       
        video = self.video_attn(video, audio)
        

        video, audio = self.coattn(video, audio)

        audiovisualfeatures = torch.cat((video, audio), -1)
  
        vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
        aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))
        #seq_outs, _ = torch.max(outs,0)
        #print(seq_outs)
        #vsequence_outs.append(vouts)
        #asequence_outs.append(aouts)
        #    #fin_audio_features.append(att_audio_features)
        #   #fin_visual_features.append(att_visual_features)
        #final_aud_feat = torch.stack(fin_audio_features)
        #final_vis_feat = torch.stack(fin_visual_features)
        #vfinal_outs = torch.stack(vsequence_outs)
        #afinal_outs = torch.stack(asequence_outs)

        return vouts.squeeze(2), aouts.squeeze(2)  #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)
