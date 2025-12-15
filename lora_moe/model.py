import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from .configuration_lora_moe import LoraMoeConfig
from .modelling_lora_moe import LoraMoeModel
from types import SimpleNamespace
llm_path={
    "DeepSeek-7B": '/root/autodl-tmp/DeepSeekR1/DeepSeek-R1-Distill-Qwen-7B'
}

def get_sinusoidal_encoding(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) *
        -(torch.log(torch.tensor(10000.0)) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LLM4RIM(nn.Module):
    def __init__(self, configs):
        super(LLM4RIM, self).__init__()
        self.token_len = configs.token_len
        self.device = f"cuda:{configs.gpu}"
        self.patch_size = 16      # e.g., 16
        self.distance_bins = 128  # e.g., 128

        self.task_num_labels = {
            'occupancy': 6,
            'gesture': 12,
            'drowsiness': 3,
            'breathing': 5
        }
        self.task_input_types = {
            'occupancy': '1d',
            'gesture': '2d',
            'drowsiness': '2d',
            'breathing': '1d'
        }

        self.base_llm = Qwen2ForCausalLM.from_pretrained(
            llm_path[configs.llm_name],
            device_map=self.device,
            torch_dtype=torch.float32
        )

        model_config = LoraMoeConfig.from_pretrained(llm_path[configs.llm_name])
        model_config.experts_rank = configs.experts_rank
        model_config.experts_scale = configs.experts_scale
        model_config.num_experts_per_tok = configs.experts_num_per_tok
        model_config.num_local_experts = configs.experts_num
        model_config.output_router_logits = True

        self.moe_model = LoraMoeModel(self.base_llm, model_config)
        self.moe_model.make_experts_trainable()
        self.moe_model.print_trainable_parameters()

        self.hidden_dim = model_config.hidden_size
        self.max_patch_len = 1024

        self.encoders = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        # Positional encoding (sinusoidal)
        self.register_buffer("fixed_pos_encoding", get_sinusoidal_encoding(
            self.max_patch_len, self.hidden_dim, device="cpu"
        ))

        for task, num_labels in self.task_num_labels.items():
            if self.task_input_types[task] == '1d':
                self.encoders[task] = nn.Sequential(
                    nn.Conv1d(in_channels=self.token_len, out_channels=self.hidden_dim, kernel_size=1)
                )
            else:
                # Linear projection for patchified [patch_size * D] → hidden_dim
                self.encoders[task] = nn.Sequential(
                    nn.Linear(self.patch_size * self.distance_bins, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU()
                )
                # self.encoders[task] = nn.Linear(self.patch_size * self.distance_bins, self.hidden_dim)

            self.task_heads[task] = nn.Sequential(
                # nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.token_len, kernel_size=1),
                # nn.AdaptiveAvgPool1d(1),
                # nn.Flatten(),
                nn.Linear(in_features=self.hidden_dim, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=num_labels)
            )

    def forward(self, x_enc, task: str):
        assert task in self.task_num_labels, f"Invalid task name: {task}"
        task_input_type = self.task_input_types[task]

        if task_input_type == '1d':
            if x_enc.dim() == 3:
                x_enc = x_enc.squeeze(-1)  # [B, T]
            means = x_enc.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = (x_enc - means) / stdev

            B, T = x_enc.shape
            patches = x_enc.unfold(dimension=1, size=self.token_len, step=self.token_len)  # [B, N, token_len]
            patches = patches.permute(0, 2, 1)  # [B, token_len, N]
            # print(f'x_enc: {x_enc.shape}, patches: {patches.shape}')
            times_embeds = self.encoders[task](patches)  # Linear proj: [B, hidden_dim, N]
            times_embeds = times_embeds.permute(0, 2, 1)  # [B, N, hidden_dim]
            # print(f'times_embeds: {times_embeds.shape}')


        else:
            # x_enc: [B, T, D，C] → UWB 时间-距离矩阵
            if x_enc.dim() == 3:
                x_enc = x_enc.unsqueeze(-1)  # [B, T, D，1]
                
            x_enc = x_enc.permute(0,3,1,2) #[B, C, T,D]
            x_enc = x_enc.mean(1,keepdim=True).detach()
            means = x_enc.mean((1, 2, 3), keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5)
            x_enc = (x_enc - means) / stdev

            B, _, T, D = x_enc.shape
            x_enc = x_enc.squeeze(1)  # [B, T, D]
            patches = x_enc.unfold(dimension=1, size=self.patch_size, step=self.patch_size)  # [B, N, P, D]
            patches = patches.flatten(start_dim=2)  # [B, N, P*D]
            times_embeds = self.encoders[task](patches)  # Linear proj: [B, N, hidden_dim]
            # print(f'times_embeds: {times_embeds.shape}')

        # 加入位置编码
        N = times_embeds.size(1)
        pos_enc = self.fixed_pos_encoding[:N, :].unsqueeze(0).to(times_embeds.device)
        # print(f'pos_enc: {pos_enc.shape}')
        times_embeds = times_embeds + pos_enc

        outputs = self.moe_model(inputs_embeds=times_embeds, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]  # [B, N, hidden_dim]
        
        pooled = outputs.mean(dim=1)
        logits = self.task_heads[task](pooled)
        logits = logits.reshape(B,1,-1)
        logits = logits.mean(dim=1)
        return logits
    
    def get_nb_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param,'ds_numel'):
                num_params = param.ds_numel

            if param.__class__.__name__ == 'Params4bit':
                num_params = num_params * 2
            
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return trainable_params, all_param
    
    def print_trainable_parameters(self):
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
        
if __name__ == '__main__':
    configs = SimpleNamespace(
        llm_name = "DeepSeek-7B",
        token_len = 16,
        gpu = 0,
        patch_size = 16,
        distance_bins = 128,
        experts_rank = 8,
        experts_scale = 1.0,
        experts_num_per_tok = 2,
        experts_num = 8
    )

    # ----------- 初始化模型 ----------
    print("Initializing model...")
    model = LLM4RIM(configs)
    model.eval()
    model = model.to(model.device)
    # 模拟输入：(batch, time, variables)
    test_cases = {
        'occupancy': torch.randn(4, 512, 1),        # 1D: [B, T, 1]
        'breathing': torch.randn(4, 512),           # 1D: [B, T]
        'gesture': torch.randn(4, 512, 128, 3),     # 2D: [B, T, D, 3]
        'drowsiness': torch.randn(4, 512, 128)      # 2D: [B, T, D]
    }
    
    # ----------- 前向推理 ------------
    with torch.no_grad():
        for task, x in test_cases.items():
            print(f"▶️ Testing task: {task} | Input shape: {x.shape}")
            try:
                x = x.to(model.device)
                y = model(x, task=task)
                print(f"✅ Output shape: {y.shape} (Expected: [B, {model.task_num_labels[task]}])")
            except Exception as e:
                print(f"❌ Error on task {task}: {e}")