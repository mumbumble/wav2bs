"""Define the diffusion models which are used as SAiD model
"""
from abc import ABC
from dataclasses import dataclass
import inspect
from typing import List, Optional, Type, Union
from diffusers import DDPMScheduler,DDIMScheduler
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from src.adpt_bias_denoiser import Adpt_Bias_Denoiser
from src.wav2vec import Wav2Vec2Model


class DIFFUSION_BIAS(nn.Module):

    def __init__(self,feature_dim,id_dim):
        """
        Initialize the model
        """
        # we only use the functions in the GPt_ADPT_LOCAL_ATTEN class, so we don't need to call the __init__ function of the GPT_ADPT_LOCAL_ATTEN class
        super().__init__()

        # set up model
        self.audio_encoder = Wav2Vec2Model.from_pretrained("./model_load/wav2vec2/")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.denoiser = Adpt_Bias_Denoiser(
            max_len= 5000, # the attention mask maximum length
            nfeats= feature_dim, # the number of features in the vertices
            latent_dim = 512,
            ff_size = 1024,
            num_layers = 6,
            num_heads = 4,
            dropout = 0.1,
            normalize_before = False,
            activation = "gelu",
            arch = "trans_dec",
            id_dim=id_dim,
            audio_encoded_dim = 768,
            return_intermediate_dec = False,
            flip_sin_to_cos = True,
            freq_shift = 0,
            mem_attn_scale = 1.0,
            tgt_attn_scale = 0.1,
            period = 30,
            no_cross = False
        )

        # set up diffusion specific initialization
        self.scheduler = DDIMScheduler(
            num_train_timesteps= 1000,
            beta_start= 0.0001,
            beta_end= 0.012,
            beta_schedule= 'scaled_linear', # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
            # variance_type: 'fixed_small'
            clip_sample= False ,# clip sample to -1~1
            prediction_type= 'sample',
            # below are for ddim
            set_alpha_to_one= False,
            steps_offset= 1)
        
        self.noise_scheduler= DDPMScheduler(num_train_timesteps= 1000,
            beta_start= 0.0001,
            beta_end= 0.012,
            beta_schedule='scaled_linear', # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
            variance_type= 'fixed_small',
            prediction_type= 'sample',
            clip_sample= False)


        # set up the hidden state resizing parameters
        self.audio_fps = 50
        self.hidden_fps = 30
        # set up the vertice dimension
        self.nfeats = feature_dim


        # guided diffusion
        self.guidance_uncondp = 0.0#cfg.model.guidance_uncondp if hasattr(cfg.model, "guidance_uncondp") else 0.0
        self.guidance_scale = 1.0 #cfg.model.guidance_scale if hasattr(cfg.model, "guidance_scale") else 1.0
        # assert self.guidance_scale >= 0.0 and self.guidance_scale <= 1.0
        assert self.guidance_scale >= 0.0 
        self.do_classifier_free_guidence = self.guidance_scale > 0.0
        self.denoiser_id_dim=id_dim

        
    def forward(self, audio, audio_mask, vertice, vertice_mask, criterion,fps,id=None):
        """
        Forward pass of the model
        """
        # get the hidden states from the audio
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, fps, frame_num=frame_num).last_hidden_state
        hidden_states = self.denoiser.audio_feature_map(hidden_states)
        
        vertice_attention = torch.ones(
            hidden_states.shape[0], 
            hidden_states.shape[1], # in our setting, the length of the vertice_attention should be the same as the length of the hidden_state
        ).long().to(hidden_states.device) # this attention should be long type

        
        vertice_input = vertice # vertice_input.shape = [batch_size, vert_len, vert_dim]
        # perform the diffusion forward process
        vertice_output = self._diffusion_process(
            vertice_input, 
            hidden_states,
            id,
            vertice_attention
        ) 
        
        active_loss = vertice_mask.view(-1) == 1
        active_logits = vertice_output.view(-1, vertice_output.size(-1))[active_loss]
        active_labels = vertice.view(-1, vertice.size(-1))[active_loss]

        rec_loss = criterion(active_logits, active_labels)
        vel_loss=criterion(active_logits[:, 1:] - active_logits[:, :-1],active_labels[:, 1:] - active_labels[:, :-1])
        loss= 1*vel_loss+rec_loss

        return loss,vel_loss,rec_loss,vertice_output
    
    def predict(self, audio,step, fps,id=None):

        audio=audio.unsqueeze(0)
        if id is not None:
            id=id.unsqueeze(0)
        hidden_states = self.audio_encoder(audio, fps).last_hidden_state
        frame_num = hidden_states.shape[1]
        hidden_states = self.denoiser.audio_feature_map(hidden_states)

        vertice_attention = torch.ones(
            hidden_states.shape[0], 
            hidden_states.shape[1], # in our setting, the length of the vertice_attention should be the same as the length of the hidden_state
        ).long().to(hidden_states.device) # this attention should be long type
        

        # perform the diffusion revise process
        vertice_output = self._diffusion_reverse(
            hidden_states,
            id,
            vertice_attention = vertice_attention,
            step=step,
        ) # vertice_output.shape = [batch_size, vert_len, vert_dim]
         
        vertice_output = self.smooth(vertice_output)
        return vertice_output

    def _diffusion_reverse(
        self,
        hidden_state: torch.Tensor,
        id,
        vertice_attention: torch.Tensor,
        step: int,
    ):  
        """
        Perform the diffusion reverse process during inference
        Args:
            hidden_state (torch.Tensor): [batch_size, seq_len, latent_dim], the audio feature, padding may included
            id (torch.Tensor): [batch_size, id_dim], the id of the subject
            vertice_attention (torch.Tensor): [batch_size, vert_len], the attention of the vertices to indicate which vertices are valid, since the audio feature has the same length as the vertices, the vertice_attention should be the same length as the hidden_state
        """

        if id is not None:
            object_emb = self.denoiser.obj_vector(torch.argmax(id, dim = 1)).unsqueeze(1)
        else:
            object_emb=None
        # sample noise
        vertices = torch.randn(
            (
                hidden_state.shape[0], # batch_size
                hidden_state.shape[1], # vert_len
                self.nfeats, # latent_dim
            ),
            device = hidden_state.device,
            dtype = torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        vertices = vertices * self.scheduler.init_noise_sigma

        # set timesteps
        self.scheduler.set_timesteps(step)#self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(hidden_state.device, non_blocking=True)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = 0.0 #self.cfg.model.scheduler.eta
            
        # perform denoising
        for i, t in enumerate(timesteps):

            # perform denoising step
            vertices_pred = self.denoiser(
                vertice_input = vertices, # vertices.shape = [batch_size, vert_len, latent_dim]
                hidden_state = hidden_state, # hidden_state.shape = [batch_size, seq_len, latent_dim]
                timesteps = t.expand(hidden_state.shape[0]), # timesteps.shape = [batch_size]
                adapter = object_emb,
                tgt_mask = self._tgt_mask(vertice_attention,object_emb), # tgt_mask.shape = [vert_len, vert_len]
                memory_mask = self._memory_mask(vertice_attention,object_emb), # memory_mask.shape = [vert_len, seq_len]
                tgt_key_padding_mask = self._tgt_key_padding_mask(vertice_attention,object_emb), # tgt_key_padding_mask.shape = [batch_size, vert_len]
                memory_key_padding_mask = self._mem_key_padding_mask(vertice_attention,object_emb), # memory_key_padding_mask.shape = [batch_size, seq_len]
            )

            vertices = self.scheduler.step(vertices_pred, t, vertices, **extra_step_kwargs).prev_sample
                
        return vertices    

    def _diffusion_process(
        self,
        vertice_input: torch.Tensor,
        hidden_state: torch.Tensor,
        id,
        vertice_attention: Optional[torch.Tensor] = None,
    ):  
        """
        Perform the diffusion forward process during training
        Args:
            vertice_input (torch.Tensor): [batch_size, vert_len, vert_dim], the grount truth vertices, padding may included
            hidden_state (torch.Tensor): [batch_size, seq_len, latent_dim], the audio feature, padding may included
            id (torch.Tensor): [batch_size, id_dim], the id of the subject
            vertice_attention (torch.Tensor): [batch_size, vert_len], the attention of the vertices to indicate which vertices are valid, since the audio feature has the same length as the vertices, the vertice_attention should be the same length as the hidden_state
        """

        # extract the id style
        if id is not None :
             object_emb = self.denoiser.obj_vector(torch.argmax(id, dim = 1)).unsqueeze(1)
        else :
            object_emb = None
        # sample noise
        noise = torch.randn_like(vertice_input) # noise.shape = [batch_size, vert_len, vert_dim]

        # sample a random timestep for the minibatch
        bsz = vertice_input.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device = vertice_input.device
        ) # timesteps.shape = [batch_size]

        # add noise to the latents
        noise_input = self.noise_scheduler.add_noise(
            vertice_input,
            noise,
            timesteps,
        ) # noise_input.shape = [batch_size, vert_len, vert_dim]

        # predict the noise or the input
        vertice_pred = self.denoiser(
            vertice_input = noise_input, # noise_input.shape = [batch_size, vert_len, vert_dim]
            hidden_state = hidden_state, # hidden_state.shape = [batch_size, seq_len, latent_dim]
            timesteps = timesteps, # timesteps.shape = [batch_size],
            adapter=object_emb,
            tgt_mask = self._tgt_mask(vertice_attention,object_emb), # tgt_mask.shape = [vert_len, vert_len]
            memory_mask = self._memory_mask(vertice_attention,object_emb), # memory_mask.shape = [vert_len, seq_len]
            tgt_key_padding_mask = self._tgt_key_padding_mask(vertice_attention,object_emb), # tgt_key_padding_mask.shape = [batch_size, vert_len]
            memory_key_padding_mask = self._mem_key_padding_mask(vertice_attention,object_emb), # memory_key_padding_mask.shape = [batch_size, seq_len]
        )

        return vertice_pred
    
    def smooth(self, vertices):
        vertices_smooth = F.avg_pool1d(
            vertices.permute(0, 2, 1),
            kernel_size=3, 
            stride=1, 
            padding=1
        ).permute(0, 2, 1)  # smooth the prediction with a moving average filter
        vertices[:, 1:-1] = vertices_smooth[:, 1:-1]
        return vertices
    
    def _memory_mask(self, hidden_attention,object_emb ):
        """
        Create memory_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, source_len]
            frame_num: int
        """
        if object_emb is None:
            mask_num=1
        else:
            mask_num=2
        if self.denoiser.use_mem_attn_bias:
            # since the source_len is the same as the target_len, we can use the same size to create the mask
            memory_mask = self.denoiser.memory_bi_bias[:hidden_attention.shape[1], :hidden_attention.shape[1]]

            # since the adapter is used, we need to unmask another position to make the adapter work, this position is the first and the second positions of the memory_mask
            adpater_mask = torch.zeros_like(memory_mask[:, :mask_num]) # [1, source_len], since the apdater length = id + time = 2
            memory_mask = torch.cat([adpater_mask, memory_mask], dim = 1) # [source_len, latent_len + 2]

            return  memory_mask.bool().to(hidden_attention.device) # [source_len, latent_len + 2]
        else:
            return None
        
    def _tgt_mask(self, vertice_attention,object_emb ):
        """
        Create tgt_key_padding_mask for transformer decoder
        Args:
            vertice_attention: [batch_len, source_len]
            frame_num: int
        """
        if object_emb is None:
            mask_num=1
        else:
            mask_num=2

        if self.denoiser.use_tgt_attn_bias:
            batch_size = vertice_attention.shape[0]
            tgt_mask = self.denoiser.target_bi_bias[:, :vertice_attention.shape[1], :vertice_attention.shape[1]] # [num_heads, target_len, target_len]
            adapter_mask = torch.zeros_like(tgt_mask[..., :mask_num]) # [num_heads, target_len, 2], since the apdater length = id + time = 2
            tgt_mask = torch.cat([adapter_mask, tgt_mask], dim = -1) # [num_heads, target_len, target_len + 2]

            # repeat the mask for each batch
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1) # [batch_size * num_heads, target_len, target_len + 2]
            return tgt_mask.to(vertice_attention.device, non_blocking=True) # [batch_size * num_heads, target_len, target_len + 2]
        else:
            return None

    def _mem_key_padding_mask(self, vertice_attention,object_emb):
        """
        Create mem_key_padding_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, source_len]
        """
        if object_emb is None:
            mask_num=1
        else:
            mask_num=2
        # since the adapter is used, we need to unmask another position to make the adapter work
        # this position is the first and the second positions of the mem_key_padding_mask
        adpater_mask = torch.ones_like(vertice_attention[:, :mask_num]) # [batch_size, 2], since the apdater length = id + time = 2
        vertice_attention = torch.cat([adpater_mask, vertice_attention], dim = 1) # [batch_size, source_len + 2]

        # mask with 1 means that the position is masked
        return ~vertice_attention.bool()
    
    def _tgt_key_padding_mask(self, vertice_attention,object_emb):
        """
        Create tgt_key_padding_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, target_len]
        """
        # since the adapter is used, we need to unmask another position to make the adapter work
        # this position is the first and the second positions of the tgt_key_padding_mask
        if object_emb is None:
            mask_num=1
        else:
            mask_num=2
        adpater_mask = torch.ones_like(vertice_attention[:, :mask_num])
        vertice_attention = torch.cat([adpater_mask, vertice_attention], dim = 1)

        # mask with 1 means that the position is masked
        return ~vertice_attention.bool()