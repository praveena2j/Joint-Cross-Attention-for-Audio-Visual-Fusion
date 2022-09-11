import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class BottomUpExtract(nn.Module):

	def __init__(self, emed_dim, dim):
		super(BottomUpExtract, self).__init__()
		self.attn = PositionAttn(emed_dim, dim)

	def forward(self, video, audio):
		feat = self.attn(video, audio)
		# feat = F.normalize(self.linear(feat), dim=-1)

		return feat


# audio-guided attention
class PositionAttn(nn.Module):

	def __init__(self, embed_dim, dim):
		super(PositionAttn, self).__init__()
		self.affine_audio = nn.Linear(embed_dim, dim)
		self.affine_video = nn.Linear(512, dim)
		self.affine_v = nn.Linear(dim, 49, bias=False)
		self.affine_g = nn.Linear(dim, 49, bias=False)
		self.affine_h = nn.Linear(49, 1, bias=False)
		self.affine_feat = nn.Linear(512, dim)
		self.relu = nn.ReLU()

	def forward(self, video, audio):
		v_t = video.view(video.size(0) * video.size(1), -1, 512).contiguous()
		V = v_t

		# Audio-guided visual attention
		v_t = self.relu(self.affine_video(v_t))
		a_t = audio.view(-1, audio.size(-1))

		a_t = self.relu(self.affine_audio(a_t))

		content_v = self.affine_v(v_t) \
					+ self.affine_g(a_t).unsqueeze(2)

		z_t = self.affine_h((torch.tanh(content_v))).squeeze(2)

		alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))  # attention map


		c_t = torch.bmm(alpha_t, V).view(-1, 512)
		video_t = c_t.view(video.size(0), -1, 512)

		video_t = self.affine_feat(video_t)

		return video_t
