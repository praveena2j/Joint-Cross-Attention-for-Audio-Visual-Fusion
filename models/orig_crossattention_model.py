import torch.nn as nn
import torch
from torch.nn import init
from layer.normalize_layer import Normalization
from layer.audio_extract_layer import LSTM, GRU
from layer.video_extract_layer import BottomUpExtract
from layer.ave_coattn_layer import DCNLayer
from layer.predict_layer import PredictLayer
import torch.nn.functional as F

class DCNWithRCNN(nn.Module):

    def __init__(self, opt):
        super(DCNWithRCNN, self).__init__()
        print("Initializing DCNWithRCNN...")
        if opt.rnn == "GRU":
            self.audio_extract = GRU(128, opt.audio_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
            self.video_extract = GRU(opt.video_size, opt.video_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
        else:
            self.audio_extract = LSTM(128, opt.audio_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
            self.video_extract = LSTM(opt.video_size, opt.video_size, opt.num_layers, opt.droprnn, residual_embeddings=True)
        self.normalize = Normalization()
        self.video_attn = BottomUpExtract(opt.audio_size, opt.video_size)
        self.coattn = DCNLayer(opt.video_size, opt.audio_size, opt.num_seq, opt.dropout)
        self.predict = PredictLayer(opt.video_size, opt.audio_size, opt.num_classes, opt.dropout)

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

    def forward(self, audio, video):
        audio, video = self.normalize(audio, video)
        audio = self.audio_extract(audio)
        video = self.video_attn(video, audio)

        video = self.video_extract(video)

        video, audio = self.coattn(video, audio)
        score = self.predict(video, audio)

        return score