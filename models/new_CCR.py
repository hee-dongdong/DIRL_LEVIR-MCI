from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LMPredictionHead(nn.Module):
    """
    Language Model Prediction Head that shares weights with word embeddings.
    """
    def __init__(self, cfg, embed_weight):
        super().__init__()
        self.hidden_size = cfg.model.transformer_decoder.att_dim
        self.vocab_size = cfg.model.transformer_decoder.vocab_size
        
        # Transform hidden states
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        
        # Tied weight for output projection
        self.decoder = nn.Linear(embed_weight.size(1), embed_weight.size(0), bias=False)
        self.decoder.weight = embed_weight
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class CrossTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        self.self_att = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.cross_att = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2, dec_mask, diagonal_mask=True):
        tgt_length = dec_mask.size(1)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        pad_mask = 1 - dec_mask
        pad_mask = pad_mask.bool()
        attn_output, self_weight = self.self_att(input1, input1, input1, attn_mask=mask,key_padding_mask=pad_mask)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)

        text = output.mean(0)

        attn_output, cross_weight = self.cross_att(output, input2, input2)
        output = output + self.dropout2(attn_output)
        output = self.norm2(output)

        visual = output.mean(0)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm3(output)

        #############################
        logit1 = (text @ visual.T) / 0.5
        logit2 = (visual @ text.T) / 0.5
        logit = (logit1 + logit2) / 2
        logpt = F.log_softmax(logit, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        ############################

        return output, cross_weight, sim_loss


class AuxiliaryDecoderWithSkip(nn.Module):
    """
    Auxiliary Decoder with skip connections from original input feature.
    The original input (x) is passed as a skip connection to each Block.
    Number of blocks is configurable via num_blocks parameter.
    
    Input: (B, hidden_size, 14, 14)
    Output: (B, num_classes, 256, 256)
    """
    def __init__(self, hidden_size, num_classes=3, num_blocks=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.input_size = 14
        self.output_size = 256
        
        # Calculate intermediate sizes for each block
        # We need to go from 14 -> 256, with num_blocks upsampling steps + final resize
        # Each block doubles the size, then final block resizes to 256
        self.sizes = self._compute_sizes()
        
        # Channel progression: start from hidden_size, decrease by half each block
        # Minimum channel size is 32
        self.channels = self._compute_channels()
        
        # Build blocks dynamically
        self.skip_adapters = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_blocks):
            # Input channels for this block
            if i == 0:
                in_channels = hidden_size
            else:
                in_channels = self.channels[i - 1]
            
            out_channels = self.channels[i]
            skip_channels = out_channels  # Skip adapter outputs same as block output
            
            # Skip adapter: adapts original input to this block's channel
            self.skip_adapters.append(nn.Sequential(
                nn.Conv2d(hidden_size, skip_channels, kernel_size=1),
                nn.BatchNorm2d(skip_channels)
            ))
            
            # Conv block: input is previous output + skip
            self.convs.append(nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_channels))
        
        # Final conv (no skip connection at final stage to save memory)
        self.conv_final = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _compute_sizes(self):
        """Compute intermediate spatial sizes for each block."""
        sizes = []
        current_size = self.input_size
        for i in range(self.num_blocks):
            current_size = current_size * 2
            sizes.append(current_size)
        return sizes
    
    def _compute_channels(self):
        """Compute channel sizes for each block. Decreases by half each block, min 32."""
        channels = []
        current_channels = 256  # Start with 256 channels
        for i in range(self.num_blocks):
            channels.append(current_channels)
            current_channels = max(32, current_channels // 2)
        return channels
    
    def forward(self, x):
        # x: (B, hidden_size, 14, 14) - original input feature
        x_orig = x  # Save original input for skip connections
        out = x
        
        for i in range(self.num_blocks):
            target_size = self.sizes[i]
            
            # Compute skip connection from original input
            if i == 0:
                # First block: no need to upsample original input
                skip = self.skip_adapters[i](x_orig)
            else:
                # Upsample original input to match current resolution
                skip = F.interpolate(x_orig, size=(target_size // 2, target_size // 2), 
                                    mode='bilinear', align_corners=True)
                skip = self.skip_adapters[i](skip)
            
            # Concatenate with skip connection
            out = torch.cat([out, skip], dim=1)
            
            # Conv -> BN -> ReLU
            out = self.relu(self.bns[i](self.convs[i](out)))
            
            # Upsample
            out = F.interpolate(out, size=(target_size, target_size), 
                               mode='bilinear', align_corners=True)
        
        # Final: resize to output_size (256x256) without skip connection (memory saving)
        out = F.interpolate(out, size=(self.output_size, self.output_size), 
                           mode='bilinear', align_corners=True)
        
        out = self.conv_final(out)  # (B, num_classes, 256, 256)
        
        return out


class MaskCrossAttention(nn.Module):
    """
    Cross-attention layer that uses predicted mask as Key and Value.
    Encodes the mask and applies cross-attention with text query.
    """
    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        
        # Mask encoder: (B, 3, 256, 256) -> (B, 196, d_model)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, d_model, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14)),  # -> 14x14
        )
        
        # Cross-attention with mask
        self.cross_att = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, query, mask_logits):
        """
        Args:
            query: (L, B, D) - text hidden states from previous layer
            mask_logits: (B, 3, 256, 256) - predicted mask logits
            
        Returns:
            output: (L, B, D) - attended output
        """
        # Encode mask
        mask_features = self.mask_encoder(mask_logits)  # (B, D, 14, 14)
        B, D, H, W = mask_features.shape
        mask_features = mask_features.view(B, D, -1).permute(2, 0, 1)  # (196, B, D)
        
        # Cross-attention: query attends to mask features
        attn_output, _ = self.cross_att(query, mask_features, mask_features)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        
        # FFN
        ffn_output = self.ffn(output)
        output = output + self.dropout2(ffn_output)
        output = self.norm2(output)
        
        return output

class Core(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.model.transformer_decoder.vocab_size
        self.word_embed_size = cfg.model.transformer_decoder.word_dim
        self.hidden_size = cfg.model.transformer_decoder.att_dim
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_size, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(self.word_embed_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
        # nn.ReLU()
        )
        self.embed_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.n_head = cfg.model.transformer_decoder.att_head

        self.position_enc = PositionEncoding(n_filters=self.word_embed_size,
                                                    max_len=cfg.model.transformer_decoder.seq_length)

        # Ablation settings
        self.use_skip_connection = cfg.model.auxiliary.use_skip_connection
        self.use_mask_in_decoder = cfg.model.auxiliary.use_mask_in_decoder
        self.num_blocks = getattr(cfg.model.auxiliary, 'num_blocks', 5)  # Default to 5 blocks

        # Auxiliary decoder (conditional based on config)
        if self.use_skip_connection:
            self.aux_decoder = AuxiliaryDecoderWithSkip(self.hidden_size, num_classes=3, num_blocks=self.num_blocks)
        else:
            self.aux_decoder = nn.Sequential(
                # 14 -> 28
                nn.Conv2d(self.hidden_size, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 28 -> 56
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 56 -> 112
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 112 -> 224
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # 224 -> 256
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
                nn.Conv2d(32, 3, kernel_size=1) # 3 classes: background, class1, class2
            )
        self.aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        # Mask cross-attention layer (optional)
        if self.use_mask_in_decoder:
            self.mask_cross_attention = MaskCrossAttention(self.hidden_size, self.n_head)

        self.num_hidden_layers = cfg.model.transformer_decoder.att_layer
        self.layer = nn.ModuleList([CrossTransformer(self.hidden_size, self.n_head)
                                    for _ in range(self.num_hidden_layers)])

    def forward(self, seq, dec_mask, diff_bef, diff_aft,
                auxiliary_target=None,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        dec_hidden_states = self.position_enc(self.embed(seq))
        dec_hidden_states = self.fc(dec_hidden_states)
        dec_hidden_states = dec_hidden_states.transpose(0,1)
        enc_outputs = torch.cat([diff_bef, diff_aft], -1)
        enc_outputs = self.embed_fc(enc_outputs)

        # Compute auxiliary mask prediction and loss
        aux_loss, aux_pred = self._compute_auxiliary_loss(enc_outputs, auxiliary_target)

        enc_outputs = enc_outputs.transpose(0, 1)

        all_encoder_layers = []
        all_att_vis = []
        all_sim_loss = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states, attention_weight, sim_loss = layer_module(
                dec_hidden_states, enc_outputs, dec_mask, diagonal_mask=diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
                all_att_vis.append(attention_weight)
                all_sim_loss.append(sim_loss)
        
        # Apply mask cross-attention if enabled and aux_pred is available
        if self.use_mask_in_decoder and aux_pred is not None:
            dec_hidden_states = self.mask_cross_attention(dec_hidden_states, aux_pred)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
            all_att_vis.append(attention_weight)
            all_sim_loss.append(sim_loss)
        return all_encoder_layers[-1].transpose(0, 1), all_att_vis[-1], all_sim_loss[-1], aux_loss

    def _compute_auxiliary_loss(self, enc_outputs, auxiliary_target):
        """
        Compute auxiliary segmentation loss and return predicted mask.
        
        Returns:
            aux_loss: segmentation loss
            aux_pred: predicted mask logits (B, 3, 256, 256) or None if no target
        """
        batch_size, num_tokens, hidden_dim = enc_outputs.shape
        spatial_size = int(round(math.sqrt(num_tokens)))
        if spatial_size * spatial_size != num_tokens:
            raise ValueError("Number of tokens does not form a square spatial map.")

        spatial_features = enc_outputs.view(batch_size, spatial_size, spatial_size, hidden_dim).permute(0, 3, 1, 2)
        aux_pred = self.aux_decoder(spatial_features) # (B, 3, 256, 256)

        if auxiliary_target is None:
            return enc_outputs.new_tensor(0.0), aux_pred

        target = auxiliary_target.to(enc_outputs.device).long() # (B, H, W)

        # Ensure target matches output size if not already
        if target.shape[-2:] != aux_pred.shape[-2:]:
             # Nearest neighbor for masks
            target = F.interpolate(target.unsqueeze(1).float(), size=aux_pred.shape[-2:], mode='nearest').squeeze(1).long()

        aux_loss = self.aux_loss_fn(aux_pred, target)
        return aux_loss, aux_pred



class CCR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.model.transformer_decoder.vocab_size
        self.word_embed_size = cfg.model.transformer_decoder.word_dim
        self.hidden_size = cfg.model.transformer_decoder.att_dim
        self.label_smoothing = cfg.model.transformer_decoder.label_smoothing

        self.seq_length = cfg.model.transformer_decoder.seq_length

        self.core = Core(cfg)

        self.share_wd_cls_weight = cfg.model.transformer_decoder.share_wd_cls_weight


        if self.share_wd_cls_weight:
            logit_weight = self.core.embed.weight

            self.logit = LMPredictionHead(cfg, logit_weight)
        else:
            self.logit = nn.Linear(self.hidden_size, self.vocab_size)


        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def _forward(self,
                 diff_bef, diff_aft, seq, mask, labels_with_ignore=None,
                 auxiliary_target=None):


        decoder_outputs, attention_weight, sim_loss, aux_loss = self.core(
            seq, mask, diff_bef, diff_aft, auxiliary_target=auxiliary_target, diagonal_mask=True)  # [:,1:,:]  # (N, Lt, D)
        prediction_scores = self.logit(decoder_outputs)
        caption_loss = 0.
        if labels_with_ignore is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.vocab_size),
                                          labels_with_ignore.view(-1))
        return caption_loss, attention_weight, sim_loss, aux_loss

    def sample(
            self,
            diff_bef, diff_aft, start_idx=2, unk_idx=1, sample_max=0):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """

        bsz = diff_bef.size(0)
        max_cap_len = self.seq_length
        text_input_ids = diff_bef.new_zeros((bsz, max_cap_len), dtype=torch.long)
        text_masks = diff_bef.new_zeros(bsz, max_cap_len).float()  # zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_cap_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            decoder_outputs, attention_weight, sim_loss, _ = self.core(text_input_ids, text_masks, diff_bef, diff_aft, auxiliary_target=None, diagonal_mask=True)
            pred_scores = F.log_softmax(self.logit(decoder_outputs), -1)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            if sample_max:
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words
            else:
                prob_prev = torch.exp(pred_scores[:, dec_idx])
                next_words = torch.multinomial(prob_prev, 1)
                next_symbols = next_words.view(-1).long()

            if dec_idx == 0:
                unfinished = next_symbols != 3 # 3 is the end sign
            else:
                unfinished = unfinished * (next_symbols != 3)
            next_symbols = next_symbols * unfinished.type_as(next_symbols)

            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return text_input_ids, attention_weight

    def get_auxiliary_mask(self, diff_bef, diff_aft):
        """
        Get the predicted auxiliary mask from the auxiliary decoder.
        
        Args:
            diff_bef: difference features before (B, N, D)
            diff_aft: difference features after (B, N, D)
            
        Returns:
            aux_pred: predicted mask logits (B, num_classes, H, W)
            aux_pred_class: predicted class indices (B, H, W)
        """
        enc_outputs = torch.cat([diff_bef, diff_aft], -1)
        enc_outputs = self.core.embed_fc(enc_outputs)
        
        batch_size, num_tokens, hidden_dim = enc_outputs.shape
        spatial_size = int(round(math.sqrt(num_tokens)))
        
        spatial_features = enc_outputs.view(batch_size, spatial_size, spatial_size, hidden_dim).permute(0, 3, 1, 2)
        aux_pred = self.core.aux_decoder(spatial_features)  # (B, 3, 256, 256)
        aux_pred_class = aux_pred.argmax(dim=1)  # (B, 256, 256)
        
        return aux_pred, aux_pred_class
