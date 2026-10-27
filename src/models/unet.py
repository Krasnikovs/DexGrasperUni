import torch
import torch.nn as nn
from typing import Dict

from utils.utils import ResBlock, SpatialTransformer
from utils.utils import timestep_embedding
from models.pointtransformer import PointTransformerEnc, PointTransformerBlock
# from einops import rearrange

class UNetModel(nn.Module):
    def __init__(self, params, device) -> None:
        super(UNetModel, self).__init__()
        self.collected_points = []
        
        self.d_x = params['d_x']
        self.d_model = params['d_model']
        self.nblocks = params['nblocks']
        self.resblock_dropout = params['resblock_dropout']
        self.transformer_num_heads = params['transformer_num_heads']
        self.transformer_dim_head = params['transformer_dim_head']
        self.transformer_dropout = params['transformer_dropout']
        self.transformer_depth = params['transformer_depth']
        self.transformer_mult_ff = params['transformer_mult_ff']
        self.context_dim = params['context_dim']
        self.use_position_embedding = params['use_position_embedding']

        self.device = device
        
        self.object_point_model = 'PointTransformer'
        self.set_scene_model()
        
        self.use_llm = False
        
        time_embed_dim = self.d_model * params['time_embed_mult']
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

    def set_scene_model(self, c: int = 3, num_points: int = 2048):
        scene_model_args = {'c': c, 'num_points': num_points}
        self.scene_model = PointTransformerEnc(PointTransformerBlock, [2, 3, 4, 6, 3], **scene_model_args).to(torch.device("cuda"))
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = torch.func.rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = torch.func.rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        if self.object_point_model == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = torch.func.rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
            if self.use_llm:
                text_embedding = self.text_model(data['text'])[:,0,:]   # Get [CLS] embedding (all x 77 x 512) -> (all x 512)
                batch_embedding = torch.split(text_embedding, data['sentence_cnt']) # tuple of splitted embedding (? x 512)
                text_feat = torch.cat([a.max(dim=0,keepdim=True)[0] for a in batch_embedding], dim=0).reshape(b,1,-1) # (b,1,512) 
                scene_feat = torch.cat([text_feat, scene_feat], dim=1)

        elif self.object_point_model == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        elif self.object_point_model == 'PointNet2':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat
