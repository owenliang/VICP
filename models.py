import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM



class SimpleQFormer(nn.Module):
    """A minimal Q-Former: learnable queries cross-attend to visual tokens.

    Inputs are expected as concatenated pair features of shape (B, hidden_size * 2).
    The module reshapes to 2 visual tokens per sample, projects to the Q-Former
    hidden size, and runs a small Transformer decoder over learnable queries with
    cross-attention to the visual tokens, returning (B, num_query_tokens, out_dim).
    """

    def __init__(self,
                 visual_token_dim: int,
                 num_visual_tokens: int,
                 num_query_tokens: int,
                 qformer_hidden_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 out_dim = None):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_visual_tokens = num_visual_tokens
        self.hidden_dim = qformer_hidden_dim
        self.out_dim = out_dim if out_dim is not None else qformer_hidden_dim

        # Project visual tokens to Q-Former hidden size
        self.visual_proj = nn.Linear(visual_token_dim, qformer_hidden_dim)

        # Learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_query_tokens, qformer_hidden_dim) * 0.02)

        # Transformer decoder layers with cross-attention over visual tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=qformer_hidden_dim,
            nhead=num_heads,
            dim_feedforward=qformer_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(qformer_hidden_dim)

        # Output projection if LM embedding dim differs
        self.output_proj = None
        if self.out_dim != qformer_hidden_dim:
            self.output_proj = nn.Linear(qformer_hidden_dim, self.out_dim)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """pair_features: (B, visual_token_dim * num_visual_tokens)
        Returns: (B, num_query_tokens, out_dim)
        """
        bsz = pair_features.size(0)

        # Recover visual tokens: (B, num_visual_tokens, visual_token_dim)
        visual_tokens = pair_features.view(bsz, self.num_visual_tokens, -1) # 一个样本是2张图
        # Project to hidden dim and switch to (S, B, D)
        visual_tokens = self.visual_proj(visual_tokens)  # (B, S, D)，将视觉emb维度（用作计算k,v）转成和query维度一样
        memory = visual_tokens.transpose(0, 1).contiguous()  # (S, B, D)

        '''
        下面就是Q,K,V的计算了，用N个Query emb去和M个Visual Embedding做交叉注意力
        '''
        # Prepare queries: (T, B, D)
        query = self.query_embeddings.unsqueeze(1).expand(-1, bsz, -1)  # (T, B, D)

        # Decoder with cross-attention
        out = self.decoder(tgt=query, memory=memory)  # (T, B, D)
        out = self.norm(out)  # (T, B, D)
        out = out.transpose(0, 1).contiguous()  # (B, T, D)

        # 如果Q-FORMER输出和LLM emb维度不对齐再转一下
        if self.output_proj is not None:
            out = self.output_proj(out)  # (B, T, out_dim)

        return out

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 视觉编码器
        encoder = torch.hub.load('facebookresearch/dinov2', args.vision_model).eval()
        encoder.requires_grad_(False)
        self.encoder = encoder

        # 视觉Tokenizer
        encoder_copy = torch.hub.load('facebookresearch/dinov2', args.vision_model).eval()
        encoder_copy.requires_grad_(False)
        self.encoder_copy = encoder_copy

        # 给视觉编码器的最后4层transformer block注入Lora微调
        from ops.lora import LoRALayerQKV
        for i, block in enumerate(self.encoder.blocks[-4:]):
            w_qkv_linear = block.attn.qkv
            block.attn.qkv = LoRALayerQKV(
                w_qkv_linear,
                r=128,
            )

        # DINO视觉编码器的Loss函数，根据DINO编码的图片embedding，让相同petid的图片embedding距离更近，不同petid的图片embedding距离更远
        from ops.losses import HardTripletLoss
        self.loss = HardTripletLoss(margin=0.1, hardest=True)

        self.num_layers = len(encoder.blocks)
        
        # Qwen LLM部分
        self.lm = AutoModelForCausalLM.from_pretrained(args.llm_model)
        self.lm.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model)

        # 视觉Tokenizer输出的patch embedding维度
        self.hidden_size = self.encoder.embed_dim
        
        # Replace MLP projector with a learnable Q-Former
        # Q-Former，用固定序列长度的的Query参数（trainable），去和视觉Tokenizer Embedding做交叉注意力
        self.mm_projector = SimpleQFormer(
            visual_token_dim=self.hidden_size,
            num_visual_tokens=2,  # 输入了几个视觉emb
            num_query_tokens=self.args.num_id_tokens, # 准备几个query emb
            qformer_hidden_dim=self.lm.config.hidden_size, # Query Embedding 的维度
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            out_dim=self.lm.config.hidden_size, 
        )
        
        # visual prompt embeddings，这里给DINO视觉编码器的最后4个transformer block各自准备了visual prompt token embedding
        self.query_embeddings = nn.Parameter(torch.randn(self.args.num_vpt_tokens * self.num_layers, self.lm.config.hidden_size) * 0.02)
        # MLP adapter to DINO视觉编码器
        self.prompt_mlp = nn.Linear(self.lm.config.hidden_size, self.hidden_size, bias=False) # LLM Emb -> VLM Emb
        self.prompt_mlp.weight.data.zero_()

    def sample_pair(self, labels):
                    # 获取样本数量和索引
        labels = labels.cpu()
        n_samples = len(labels)
        indices = torch.arange(n_samples)

        # 构建索引对 (i, j)，避免重复和自身配对
        i_idx, j_idx = torch.triu_indices(n_samples, n_samples, offset=1)

        # 比较标签，确定正负样本对
        is_positive = labels[i_idx] == labels[j_idx]

        # 分别获取正负样本对的索引
        positive_pairs = torch.stack((i_idx[is_positive], j_idx[is_positive]), dim=1)
        negative_pairs = torch.stack((i_idx[~is_positive], j_idx[~is_positive]), dim=1)

        # 确保正负样本对数量相等
        # min_pairs = min(len(positive_pairs), len(negative_pairs))
        min_pairs = min(128, len(positive_pairs), len(negative_pairs))

        # 随机采样（确保正负样本数量一致）
        positive_sampled = positive_pairs[torch.randperm(len(positive_pairs))[:min_pairs]]
        negative_sampled = negative_pairs[torch.randperm(len(negative_pairs))[:min_pairs]]

        # 为正负样本对分配标签（正样本对为1，负样本对为0）
        positive_labels = torch.ones(min_pairs, dtype=torch.long)
        negative_labels = torch.zeros(min_pairs, dtype=torch.long)

        # 合并正负样本对及其标签
        all_pairs = torch.cat((positive_sampled, negative_sampled), dim=0)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)
        return all_pairs, all_labels


    def forward(self,
                image_crops,
                labels=None,
                prompts=None,
                ):
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size() # （batch，多张，通道，高，宽）
            image_crops = image_crops.reshape(-1, nc, h, w)  # （batch*多张，通道，高，宽）

        icl_loss = torch.tensor(0.0) # in-context-learning loss

        # 256张图片，互相做对比，用来作为DINO的visual prompt
        if labels is not None and prompts is None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1) # (batch,)->(batch,1)->(batch,多张)->(batch*多张)，每个样本是同一个petid，同批次内是同一类
            # clip_image_crops = clip_image_crops.reshape(-1, nc, clip_image_crops.size(-2), clip_image_crops.size(-1))
            
            # ========== 构建ICL（In-Context Learning）样本与提示 ==========
            # 目标：让LLM通过"正负样本对+yes/no标签"学习ID判别规则，生成VPT提示引导视觉编码器
            # 1. 取前256张图（控制计算成本，同时保证足够的同类/异类组合多样性）
            # 2. 用冻结的encoder_copy提取特征作为ICL示例的视觉输入
            # 3. 随机构造正样本对（x=1, 同ID两张图+yes）与负样本对（x=0, 不同ID+yes/no取决于实际标签）
            # 4. 将图像对特征与yes/no token喂给LLM，计算ICL loss并生成视觉提示prompts
            num_examples = 256
            clip_image_crops_e = image_crops[:num_examples]
            labels_e = labels[:num_examples]
            with torch.no_grad():
                # DINO Tokenizer输入每张图片（c,h,w），输出(hidden_size）对应DINO [CLS] token的编码结果
                image_features = self.encoder_copy.forward_features(clip_image_crops_e)['x_norm_clstoken'] 

            input_ids = []
            new_image_features = []
            yes_token_id = self.tokenizer.convert_tokens_to_ids('yes')
            no_token_id = self.tokenizer.convert_tokens_to_ids('no')
            tokenmaps = {1: yes_token_id, 0: no_token_id}
            for i in range(self.args.num_icl_bs):
                s = [] # 2个image token占位符 + yes/no token
                for j in range(self.args.num_icl_samples):
                    s.extend([-1] * self.args.num_id_tokens) # num_id_tokens是QFormer的Query序列长度，代替每张图片占位token
                    x = torch.randint(0, 2, (1,)).item()
                    if x == 1:
                        idx = torch.randint(0, image_features.size(0) // 2, (1,)).item()  # 256张图其实每2张是同一个pet，这里随机取1个pet
                        s.append(tokenmaps[x]) # 图pair的yes
                        new_image_features.append(image_features.reshape(-1, 2, self.hidden_size)[idx]) # 图pair的2张img feature -- 取一个pet的两个图，（2,hidden_size）
                    elif x == 0: 
                        i1 = torch.randint(0, image_features.size(0), (1,)).item()  # 随机取一张图
                        i2 = torch.randint(0, image_features.size(0), (1,)).item()  # 随机取一张图
                        new_image_features.append(torch.stack([image_features[i1], image_features[i2]]))
                        s.append(tokenmaps[int(labels_e[i1] == labels_e[i2])])  # yes or no，随机出现
                        # s.append(int(labels_e[i1] == labels_e[i2]))
                    else:
                        raise ValueError
                input_ids.append(s)
            input_ids = torch.tensor(input_ids).cuda().long()
            
            # 要预测的目标token
            input_labels = input_ids.clone()
            input_labels[input_ids < 0] = -100 # 忽略img token，只计算yes/no token loss

            # 输入token，将img token占位符替换为token id 0就行，稍后会直接给img换成emb，不会使用token id 0
            selected = input_ids == -1
            input_ids[input_ids < 0] = 0
            
            # 所有样本的2图特征构成batch
            image_features = torch.cat(new_image_features)
            image_features = image_features.reshape(-1, self.hidden_size * 2)
            
            # 做Qformer将每个2图特征转成LLM的token emb
            # Q-Former returns (B, num_id_tokens, lm_word_emb_dim)
            image_features = self.mm_projector(image_features)
            # 手动执行LLM的embedding layer拿到token emb
            input_embeddings = self.lm.get_input_embeddings()(input_ids).clone() # （batch,seq_len,emb_dim）
            
            image_features = image_features.reshape(-1, self.lm.config.hidden_size)
            # 注意保持计算图，所以*0来实现visual token的emb清空和修改成真实visual emb
            input_embeddings[selected] = input_embeddings[selected] * 0 + image_features.to(input_embeddings.dtype)
            
            # 给LLM的transformer直接输入embedding
            outputs = self.lm(inputs_embeds=input_embeddings, labels=input_labels, use_cache=False)

            # 拿到LLM的yes/no token loss
            icl_loss = outputs.loss

            # 将visual prompts embedding参数添加到input embeddings序列的末尾
            input_embeddings2 = torch.cat([input_embeddings, self.query_embeddings.unsqueeze(0).expand(input_embeddings.size(0), -1, -1)], dim=1)
            # 再算一波，只需要拿到最后一层transformer输出的hidden state，不需要做cross entropy loss
            outputs2 = self.lm(inputs_embeds=input_embeddings2, use_cache=False, output_hidden_states=True)
            # 拿出最后一层transformer输出的visual prompts token的hidden state，注意这里visual prompts token是4倍长度，给视觉编码器最后4层分别使用
            prompts = outputs2.hidden_states[-1][:, -self.args.num_vpt_tokens * self.num_layers:]
            prompts = self.prompt_mlp(prompts) # dim维度从LLM适配到VLM


        ################ 
        # ========== 第二阶段：将LLM生成的视觉提示注入DINO编码器，完成细粒度特征提取 ==========
        # ~~~~~~~~~~~~全量图片过DINO，随机使用visual prompt
        ot_loss = torch.tensor(0.0)

        # 1. 重整prompts形状：将LLM输出的提示序列按「层数」分组
        # 原始形状 (num_icl_bs, num_vpt_tokens * num_layers, emb_dim)
        # 目标形状 (num_icl_bs, num_layers, num_vpt_tokens, emb_dim)
        # 每个样本都有完整的4层（num_layers）提示，每层有num_vpt_tokens个token
        prompts = prompts.reshape(prompts.size(0), self.num_layers, self.args.num_vpt_tokens, -1)

       # 2. 随机采样提示：为当前batch中的每张图随机分配一个提示样本（数据增强策略）
        # 【关键】prompt学到的是"ID判别规则"而非特定样本信息，因此可以跨样本复用：
        #   - 每个prompt由64对随机图像对训练得到，编码的是通用的判别策略（如关注纹理/忽略姿态）
        #   - 8个prompt彼此独立生成，已有足够多样性
        #   - 随机分配让每张图接触不同的策略变体，增强泛化、避免过拟合
        # prompts原本只有num_icl_bs个样本（如8个），但当前batch可能有更多图（如256张）
        # 通过随机索引重复采样，让每张图都能获得一个提示（可能重复使用同一个提示）
        prompts = prompts[torch.randint(0, prompts.size(0), (image_crops.size(0),))]

        # 3. 初始化DINO编码器的token序列（patch embeddings + cls token）
        x = self.encoder.prepare_tokens_with_masks(image_crops, None) # 输入单张图片，由DINO增加CLS以及转embeding表达
        
        # 4. 前向传播：分两阶段处理Transformer块
        # 阶段1：前N-4层正常前向（不注入提示）
        for blk in self.encoder.blocks[:-self.num_layers]:
            x = blk(x) # （batch, seq_len, emb_dim）
        
        # 阶段2：后4层逐层注入对应的视觉提示token
        for i, blk in enumerate(self.encoder.blocks[-self.num_layers:]):
            prompts_ = prompts[:, i]  # 取出当前Layer对应的visual prompts： (batch, num_vpt_tokens, emb_dim)
            
            # 在cls token后、patch tokens前插入视觉提示
            # 序列结构变为：[cls_token, vpt_tokens..., patch_tokens...]
            if i == 0:
                # 第一次注入：保留所有patch tokens (x[:, 1:])
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1:]], dim=1) # 在DINO transformer block输出序列的头部插入visual prompts
            else:
                # 后续注入：需要移除上一层的vpt tokens，只保留新的patch tokens
                # x[:, 1+prompts_.size(1):] 跳过cls和旧vpt，取干净的patch tokens
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1+prompts_.size(1):]], dim=1)
            
            x = blk(x)  # 通过当前Transformer块
        
        # 5. 提取最终特征
        x = self.encoder.norm(x)  # LayerNorm归一化
        patch_features = x[:, 1+prompts_.size(1):]  # 提取CLS和Visual Prompt之后的原始DINO Token Embedding
        x = x[:, 0]  # 提取cls token作为图片整体特征（用于ID判别损失）

        x = F.normalize(x, dim=-1)
        std = x.std(dim=0).mean()
        if labels is not None:
            id_loss = self.loss(x, labels) # 核心LOSS：输入全量图片的DINO输出CLS Emb、全量图片的petid，优化目标是让相同petid的emb距离更小，不同petid的emb距离更大
            
            # 这是OT LOSS，需要看下
            from ops.wpa import compute_wpa
            all_pairs, all_labels = self.sample_pair(labels)
            ot_loss = compute_wpa(patch_features[all_pairs[:, 0]], patch_features[all_pairs[:, 1]], all_labels.cuda())

        else:
            id_loss = torch.tensor(0.0)

        # ICL(in-context-learning) LOSS：LLM的yes/no Loss
        # ID LOSS：DINO的cls token Loss，即DINO的ID判别损失
        # OT LOSS：DINO的WPA对齐损失，即DINO的细粒度特征对齐损失
        loss = icl_loss * self.args.icl_loss_weight + id_loss + ot_loss * self.args.ot_loss_weight

        outputs = {
            'loss': loss,
            'id_loss': id_loss,
            'ot_loss': ot_loss,
            'icl_loss': icl_loss,
            'features': x,
            'prompts': prompts,
            'std': std,
        }

        return outputs

        
if __name__ == '__main__':
    import easydict
    EasyDict = easydict.EasyDict
    args = EasyDict()
    # args.vision_model = 'dinov2_vits14'
    args.vision_model = 'dinov2_vitb14'
    args.llm_model = 'Qwen/Qwen3-0.6B'
    args.num_id_tokens = 4
    args.num_vpt_tokens = 2
    args.num_icl_samples = 64
    args.num_icl_bs = 8
    args.icl_loss_weight = 0.0
    args.ot_loss_weight = 0.0
    model = Model(args).cuda()
    image_crops = torch.randn(2, 2, 3, 224, 224).cuda()
    labels = torch.randint(0, 10, (2,)).cuda()
    prompts = None
    outputs = model(image_crops, labels, prompts)
    # print(outputs)
    exit(0)