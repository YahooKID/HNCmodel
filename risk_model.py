from transformers import BertTokenizer
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
import torch
from torch import nn
import contextlib
import models_vit2
from util.pos_embed import interpolate_pos_embed
import torch.nn.functional as F


class risk_model(nn.Module):
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    @staticmethod
    def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @staticmethod
    def init_tokenizer(truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    
    def __init__(self, device="cpu", num_heads=12, attn_drop=0., load_cp=True, ct_pre_path=None, pet_pre_path=None):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.num_heads = num_heads
        self.Qformer, self.query_tokens = self.init_Qformer(32, 768)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer = self.Qformer.to(device)
        self.scale = (768 // self.num_heads) ** (-0.5)
        self.ct_ln_vision = LayerNorm(768)
        self.pet_ln_vision = LayerNorm(768)
        self.proj_layer = LayerNorm(768)
        self.ct_vision_encoder = models_vit2.__dict__["vit_base_patch16"](
            num_classes=0,
            drop_path_rate=0.0,
            global_pool=True,
        )
        self.pet_vision_encoder = models_vit2.__dict__["vit_base_patch16"](
            num_classes=0,
            drop_path_rate=0.0,
            global_pool=True,
        )
        if load_cp:
            assert ct_pre_path is not None
            assert pre_pre_path is not None
            #load ct_vision_encoder
            checkpoint = torch.load(ct_pre_path, map_location='cpu')
        
            checkpoint_model = checkpoint
            ct_state_dict = self.ct_vision_encoder.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != ct_state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
                    
            # interpolate_pos_embed(self.ct_vision_encoder, checkpoint_model)
            msg = self.ct_vision_encoder.load_state_dict(checkpoint_model, strict=False) 
            print(msg) 

            #load pet_vision_encoder
            checkpoint = torch.load(pet_pre_path, map_location='cpu')
            checkpoint_model = checkpoint
            pet_state_dict = self.pet_vision_encoder.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != pet_state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
                    
            # interpolate_pos_embed(self.ct_vision_encoder, checkpoint_model)
            msg = self.ct_vision_encoder.load_state_dict(checkpoint_model, strict=False) 
            print(msg)
        
        #freeze vision encoder
        for name, param in self.ct_vision_encoder.named_parameters():
            param.requires_grad = False
        self.ct_vision_encoder = self.ct_vision_encoder.eval()
        self.ct_vision_encoder.train = disabled_train
        
        for name, param in self.pet_vision_encoder.named_parameters():
            param.requires_grad = False
        self.pet_vision_encoder = self.pet_vision_encoder.eval()
        self.pet_vision_encoder.train = disabled_train
        print("freeze vision encoder")
        
        qformer_state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(qformer_state_dict[key_orig])
        
        self.qk = nn.Linear(768, 768 * 2)
        self.v = nn.Linear(768, 768)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(768, 768)
        
        self.fc_1 = nn.Linear(768, 1)
        self.fc_2 =  nn.Linear(32, 1)
        self.at1 = nn.Sigmoid()
        # self.fc_3 = nn.Linear(196 * 768, 196)
        # self.fc_4 = nn.Linear(196 ,1)
        # self.at2 = nn.ReLU()
        self.device = device
        for name, param in self.Qformer.named_parameters():
            param.requires_grad=False
        
    
    def forward(self, ctimg, petimg, texts, time, gt):
        bs = ctimg.shape[0]
        
        # x0 = self.fc_3(img.reshape(bs, -1))
        # x0 = self.at2(x0)
        # x0 = self.fc_4(x0)
        text_Qformer = self.tokenizer(
                texts,
                padding='longest',
                truncation=True,
                max_length=40,
                return_tensors="pt",
            )
        # with self.maybe_autocast():
        ct_img_embeds = self.ct_ln_vision(self.ct_vision_encoder.forward_features(ctimg))
        pet_img_embeds = self.pet_ln_vision(self.pet_vision_encoder.forward_features(petimg))
        B, N, C = ct_img_embeds.shape
        qk = self.qk(ct_img_embeds).reshape(B, N, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(pet_img_embeds).reshape(B, N, 1, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        v = v[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        img_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)
        img_embeds = self.proj(img_embeds)
        img_embeds = self.proj_layer(img_embeds)
        img_embeds = F.normalize(img_embeds)
        img_atts = torch.ones(img_embeds.shape[:-1], dtype=torch.long).to(ctimg.device)
        query_tokens = self.query_tokens.expand(img_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long).to(ctimg.device)
        Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask.to(ctimg.device)], dim=1)
        query_output = self.Qformer.bert(
            text_Qformer.input_ids.to(ctimg.device), 
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        lhs = query_output.last_hidden_state[:,:query_tokens.size(1),:]
        lhs = self.fc_1(lhs).transpose(-2, -1)
        lhs = self.at1(lhs)
        lhs = self.fc_2(lhs).reshape(B, 1)
        
        # return torch.sigmoid(torch.cat([x0, lhs], axis=1))
        return lhs
        
        

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, pred, target):
        return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCELoss = CustomBCELoss()

    def forward(self, inputss, targets):
        inputs = F.sigmoid(inputss)
        BCE_loss = self.BCELoss(inputs, targets)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * loss
        return F_loss.mean()

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()
    
    def forward(self, survtime, censor, hazard_pred):
        survtime = survtime.reshape(-1)
        i_ = survtime[:, None]
        j_ = survtime[None, :]
        R_mat = i_ <= j_
        R_mat.to(torch.float32).to(hazard_pred)
        theta = hazard_pred.reshape(-1)
        exp_theta = F.sigmoid(theta)
        loss = -torch.mean((torch.log(exp_theta) - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss

class Loss(nn.Module):
    def __init__(self, a=1, b=0):
        super(Loss, self).__init__()
        self.focalloss = CustomBCELoss()
        self.coxloss = CoxLoss()
        self.a = a
        self.b = b
    
    def forward(self, survtime, inputs, targets):
        return self.a * self.focalloss(F.sigmoid(inputs), targets) + self.b * self.coxloss(survtime, targets, inputs)
        
        