import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange

from models.swin_transformer import SwinTransformerr

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        print(q.shape)
        k = self.c_k(x)
        print(k.shape)
        v = self.c_v(x)
        print(v.shape)
        exit()

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x
    
# class MultiHeadDualBlock(nn.Module):
#     def __init__(self, dim, drop=0.1):
#         super().__init__()
#         self.c_q = nn.Linear(dim, dim)
#         # self.c_k = nn.Linear(dim, dim)
#         self.c_v = nn.Linear(dim, dim)
#         self.norm_fact = dim ** -0.5
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj_drop = nn.Dropout(drop)

#     def forward(self, x):
#         _x = x
#         B, C, N = x.shape
#         q = self.c_q(x)
#         k = self.c_k(x)
#         v = self.c_v(x)

#         attn = q @ k.transpose(-2, -1) * self.norm_fact
#         attn = self.softmax(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, C, N)
#         x = self.proj_drop(x)
#         x = x + _x
#         return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class AMBQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        # print('patches_resolution',self.patches_resolution,'depth:',depths,'numheads:',num_heads,'embeddim:',embed_dim,'windowsize',window_size,'dimmlp:',dim_mlp,'scale',scale)
        
        # self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        # self.save_output = SaveOutput()
        # hook_handles = []
        # for layer in self.vit.modules():
        #     if isinstance(layer, Block):
        #         handle = layer.register_forward_hook(self.save_output)
        #         hook_handles.append(handle)
                
        # self.swintimm= timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        
        # self.swintimm= timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        self.swintimm=timm.create_model('swin_large_patch4_window7_224_in22k',pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.swintimm.modules():
            
            if isinstance(layer, SwinTransformerBlock):                
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
                
        self.swintransformerhana=self.backbone=SwinTransformerr(hidden_dim=96,layers=(2,2,6,2),heads=(3,6,12,24),channels=3,num_classes=1,head_dim=32,window_size=7,downscaling_factors=(2,1,1,1),relative_pos_embedding=True)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(3840)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(3840, 3, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        # self.fc_score = nn.Sequential(
        #     # nn.Linear(1440, 1440),
        #     nn.ReLU(),
        #     # nn.Linear(2880,1440),
        #     # nn.ReLU(),
        #     nn.Linear(1440,720),
        #     nn.ReLU(),            
        #     nn.Dropout(0.1),
        #     nn.Linear(720, 1)
        #     # nn.Sigmoid()
        #     # nn.ReLU()
        # )
        # self.fc_weight = nn.Sequential(
        #     # nn.Linear(1440, 1440),
        #     nn.ReLU(),
        #     # nn.Linear(2880,1440),
        #     # nn.ReLU(),
        #     nn.Linear(1440,720),
        #     nn.ReLU(),            
        #     nn.Dropout(0.1),
        #     nn.Linear(720, 1)
        #     # nn.Sigmoid()
        #     # nn.ReLU()
        # )
        
        self.fc_score = nn.Sequential(
            nn.Linear(1440,1440),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(1440, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(1440,1440),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(1440, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output):
        # print(len(save_output.outputs))
        # exit()
        x6 = save_output.outputs[6][:, 1:]
        # print('x6_size',x6.shape)
        x7 = save_output.outputs[7][:, 1:]
        # print('x7_size',x7.shape)
        x16 = save_output.outputs[16][:, 1:]
        # print('x8_size',x16.shape)
        x17 = save_output.outputs[17][:, 1:]
        # print('x9_size',x17.shape)
        x20 = save_output.outputs[20][:, 1:]
        # print('x9_size',x20.shape)
        x = torch.cat((x6, x7, x16, x17, x20), dim=2)
        # print('lastcat_shape:',x.shape)
        # exit()
        return x

    def forward(self, x):
        # _x = self.vit(x)
        # x = self.extract_feature(self.save_output)
        _x = self.swintimm(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        
        print(x.shape)
        exit()
        
        x=torch.unsqueeze(x,1)        
        x=torch.cat((x,torch.zeros(x.shape[0],1,1,x.shape[-1]).cuda()),dim=2)
        x=torch.squeeze(x,1)       
        
        # Feature-wise transpose attention
        for tab in self.tablock1:
            x = tab(x)
            
        #rearranging for first stage swin
        x_conv = rearrange(x, 'b (h w) c-> b c h w', h=14, w=14)
        x_conv=self.conv1(x_conv)
        
        s1,s2,s3,s4,st,op = self.swintransformerhana(x_conv)   
        
        layers=torch.cat((s1,s2,s3,s4),dim=1)
        
        # layers_mean=layers.mean(dim=[2, 3])   
        layers_mean= rearrange(layers, 'b c h w -> b (h w) c', h=7, w=7)
        # print(layers_mean.shape)
        # exit()
        
            
        # print(s1.shape,s2.shape,s3.shape,s4.shape,st.shape,op.shape)
        
        score = torch.tensor([]).cuda()
        for i in range(layers_mean.shape[0]):
            # print('i:',x[i].shape)
            f = self.fc_score(layers_mean[i])
            w = self.fc_weight(layers_mean[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
            
        print('score',score)
        exit()
        return score
