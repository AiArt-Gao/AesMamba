from .vmamba import VSSM
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import BertModel, BertTokenizer
import warnings
from transformers.modeling_utils import ModuleUtilsMixin
from .textInjectModule import textInjectModule
from typing import Optional, Callable, List
import numpy as np
import torch.nn.functional as F

class AesMamba_p(nn.Module):
    def __init__(self, device):
        super(AesMamba_p, self).__init__()

        self.img_feature = VSSM(num_classes=0)
        d = torch.load('/data/yuhao/pretrain_model/vmamba/vmamba_tiny_e292.pth', map_location='cpu')
        print(self.img_feature.load_state_dict(d['model'], strict=False))
        self.pred_head = pred_head(768)
        self.text_backbone = bert_feature(device=device)

        self.attrInject = textInjectModule()

        self.aesthetic_loss = Bal_CE_loss(cls_num=[97, 1784, 21891, 79165, 81494, 12562, 369])

    def forward(self, img, text):
        img_feature = self.img_feature(img)
        text_feature = self.text_backbone(text)
        img_feature = self.attrInject(img_feature, text_feature)
        multi_attr_pred, pred_attr_class = self.pred_head(img_feature)
        return multi_attr_pred, pred_attr_class

    def get_loss(self, pred_attr_class, attr_class, device):
        loss = self.aesthetic_loss(pred_attr_class, attr_class.to(device))
        return loss


class pred_head(nn.Module):
    def __init__(self, dim):
        super(pred_head, self).__init__()
        self.aesthetic_head = attr_pred_head(dim, 7)

    def forward(self, feature):
        aesthetic, aesthetic_classes = self.aesthetic_head(feature)
        return aesthetic, aesthetic_classes


class attr_pred_head(nn.Module):
    def __init__(self, dim, num_classes):
        super(attr_pred_head, self).__init__()
        self.adatper = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

        self.classes_heads = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.apply(self._init_weights)

    def forward(self, feature):
        feature = feature + self.adatper(feature)
        y_pred = self.heads(feature)
        return y_pred, self.classes_heads(feature)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the models parameters
        no nn.Embedding found in the any of the models parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class bert_feature(nn.Module):
    def __init__(self, device):
        # 3090
        # checkpoint = '/data2/yuhao/pretrain_model/bert'
        # new 3090
        checkpoint = '/data/yuhao/pretrain_model/bert'

        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.bert_model = BertModel.from_pretrained(checkpoint)
        # self.config = self.bert_model.base_model.config
        # self.embeddings = copy.deepcopy(self.bert_model.base_model.embeddings)
        # self.encoder = copy.deepcopy(self.bert_model.base_model.encoder.layer[:6])

    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # text_features = self.get_text_features(**tokens)
        text_features = self.bert_model(**tokens)
        text_features = text_features.last_hidden_state
        return text_features

    def get_text_features(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, dtype=attention_mask.dtype)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        feature = embedding_output
        for i, bert_layer in enumerate(self.encoder):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            feature = bert_layer(feature,
                                 attention_mask=extended_attention_mask,
                                 head_mask=layer_head_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_extended_attention_mask,
                                 past_key_value=past_key_value,
                                 output_attentions=output_attentions,
                                 )[0]

        return feature

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the models.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the models is a decoder, apply a causal mask in addition to the padding mask
            # - if the models is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(dtype).min
        # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the models.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
    def __init__(self, cls_num=None, bal_tau=1.0):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(cls_num)
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = bal_tau * prior

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior.to(x.device).repeat((x.size(0), 1))
        x = x + prior
        x = F.log_softmax(x, dim=-1)
        target = F.one_hot(target, num_classes=7)
        loss = torch.sum(-target * x, dim=-1)
        return loss.mean()