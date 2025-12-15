# train_multitask.py
# ------------------------------------------------------------
# Multi-task trainer for LLM4RIM with Dynamic Weight Averaging
# ------------------------------------------------------------
import os
import math
import time
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# from your_module_containing_llm4rim import LLM4RIM
import argparse
from lora_moe.model import LLM4RIM 

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, task):
        self.data = x_data
        self.label = y_data
        self.task = task

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.task
    
# Optional: simple default collate (keeps shapes as-is, stacks along batch)
def default_collate(batch):
    xs, ys = zip(*batch)
    # xs are variable shapes depending on task, but all items in a given batch must match
    xs = torch.stack([torch.as_tensor(x) for x in xs], dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

# -----------------------
# DWA (Dynamic Weight Averaging)
# -----------------------
class DWA:
    """
    Implements the DWA weighting scheme:
      w_i(t) = T^{ L_i(t-1) / (L_i(t-2) + eps) }
      normalized across tasks to sum to K (num_tasks)
    For the first two epochs, equal weights are used.
    """
    def __init__(self, task_names: List[str], temperature: float = 2.0, eps: float = 1e-12):
        self.task_names = task_names
        self.T = temperature
        self.eps = eps
        # Keep per-epoch average loss history for each task
        self.loss_hist: Dict[str, List[float]] = {t: [] for t in task_names}

    def update(self, epoch_avgs: Dict[str, float]) -> Dict[str, float]:
        # record the new averages
        for t in self.task_names:
            self.loss_hist[t].append(float(epoch_avgs[t]))

        k = len(self.task_names)
        # First two epochs → uniform weights
        any_short = any(len(self.loss_hist[t]) < 3 for t in self.task_names)
        if any_short:
            return {t: 1.0 for t in self.task_names}

        # Compute unnormalized weights
        unnorm = {}
        for t in self.task_names:
            L_t_1 = self.loss_hist[t][-1]      # current epoch avg
            L_t_2 = self.loss_hist[t][-2]      # prior epoch avg
            ratio = L_t_1 / (L_t_2 + self.eps)
            unnorm[t] = self.T ** ratio

        s = sum(unnorm.values()) + self.eps
        # Normalize to sum to number of tasks (as in DWA paper)
        weights = {t: (k * unnorm[t] / s) for t in self.task_names}
        return weights

# -----------------------
# Utility: accuracy
# -----------------------
@torch.no_grad()
def top1_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

# -----------------------
# Trainer
# -----------------------
def set_module_trainable(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def print_trainable_summary(model: nn.Module, prefix=""):
    t, a = 0, 0
    for n, p in model.named_parameters():
        a += p.numel()
        if p.requires_grad: t += p.numel()
    print(f"{prefix}trainable params: {t:,} / {a:,} ({100*t/max(a,1):.2f}%)")

class TaskSpec:
    name: str
    train_ds: Dataset
    val_ds: Optional[Dataset]
    batch_size: int
    num_workers: int = 2
    pin_memory: bool = True
    collate_fn: Optional[callable] = default_collate

class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        task_specs: List[TaskSpec],
        lr: float = 2e-4,
        wd: float = 0.01,
        max_epochs: int = 10,
        ckpt_dir: str = "./checkpoints",
        grad_clip: float = 1.0,
        log_every: int = 20,
        device: str = "cuda:0",
    ):
        self.model = model.to(device)
        self.task_specs = {ts.name: ts for ts in task_specs}
        self.tasks = list(self.task_specs.keys())
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_every = log_every

        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

        # one CE loss per task (you can customize per-task if needed)
        self.criteria = {t: nn.CrossEntropyLoss() for t in self.tasks}

        # Build loaders (independent datasets)
        self.train_loaders = {
            t: DataLoader(
                self.task_specs[t].train_ds,
                batch_size=self.task_specs[t].batch_size,
                shuffle=True,
                num_workers=self.task_specs[t].num_workers,
                pin_memory=self.task_specs[t].pin_memory,
                collate_fn=self.task_specs[t].collate_fn,
                drop_last=True,
            )
            for t in self.tasks
        }
        self.val_loaders = {
            t: (None if self.task_specs[t].val_ds is None else DataLoader(
                self.task_specs[t].val_ds,
                batch_size=self.task_specs[t].batch_size,
                shuffle=False,
                num_workers=self.task_specs[t].num_workers,
                pin_memory=self.task_specs[t].pin_memory,
                collate_fn=self.task_specs[t].collate_fn,
            ))
            for t in self.tasks
        }

        # Optimizer — fine-tune experts & heads (rest may be frozen by LoraMoe)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scaler = torch.amp.GradScaler(device=device)

        # DWA
        self.dwa = DWA(self.tasks, temperature=2.0)

        # Bookkeeping
        self.best_val: Dict[str, float] = {t: 0.0 for t in self.tasks}  # best per-task acc
        self.start_epoch = 1

    def save(self, epoch: int, dwa_weights: Dict[str, float]):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val": self.best_val,
            "dwa_weights": dwa_weights,
        }
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, path)
        with open(os.path.join(self.ckpt_dir, "last.json"), "w") as f:
            json.dump({"last": path}, f)
        return path

    def train(self):
        print(f"Training on tasks: {self.tasks}")
        # epoch 1: uniform DWA weights
        dwa_weights = {t: 1.0 for t in self.tasks}

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.max_epochs} =====")
            train_log, epoch_avgs = self._train_one_epoch(epoch, dwa_weights)
            print(f"Epoch {epoch} train (per-task):")
            for t in self.tasks:
                print(f"  - {t}: loss={train_log[t]['loss_avg']:.4f}, acc={train_log[t]['acc_avg']:.4f}")

            # Update DWA weights using epoch averages
            dwa_weights = self.dwa.update(epoch_avgs)
            print("DWA weights for next epoch:", {k: round(v, 3) for k, v in dwa_weights.items()})

            # Validation
            val_log = self.validate()
            print(f"Epoch {epoch} val (per-task):")
            for t in self.tasks:
                if val_log[t] is None:
                    print(f"  - {t}: (no val set)")
                    continue
                print(f"  - {t}: loss={val_log[t]['loss_avg']:.4f}, acc={val_log[t]['acc_avg']:.4f}")

                # Track and save best-per-task
                if val_log[t]["acc_avg"] > self.best_val[t]:
                    self.best_val[t] = val_log[t]["acc_avg"]
                    best_path = self.save(epoch, dwa_weights)
                    print(f"  ✓ New best for {t}: {self.best_val[t]:.4f} — saved to {best_path}")

            # Also save a regular checkpoint each epoch
            self.save(epoch, dwa_weights)

    def _train_one_epoch(self, epoch: int, dwa_weights: Dict[str, float]):
        self.model.train()
        # Create fresh iterators for each task each epoch
        iters = {t: iter(self.train_loaders[t]) for t in self.tasks}
        num_steps = min(len(dl) for dl in self.train_loaders.values())  # balanced epoch length

        # Running sums for averages
        loss_sums = {t: 0.0 for t in self.tasks}
        acc_sums = {t: 0.0 for t in self.tasks}
        count = {t: 0 for t in self.tasks}

        step = 0
        while step < num_steps:
            step += 1
            self.optimizer.zero_grad(set_to_none=True)

            # Accumulate weighted losses across tasks in the same step
            total_loss = 0.0
            log_this_step = {}

            for t in self.tasks:
                try:
                    x, y = next(iters[t])
                except StopIteration:
                    # Re-start if a loader exhausts early (shouldn't happen with drop_last=True)
                    iters[t] = iter(self.train_loaders[t])
                    x, y = next(iters[t])

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device):
                    logits = self.model(x, task=t)  # [B, num_classes]
                    loss = self.criteria[t](logits, y)

                # record for per-task metrics
                with torch.no_grad():
                    acc = top1_acc(logits, y)

                weight = dwa_weights.get(t, 1.0)
                weighted_loss = loss * weight
                total_loss = total_loss + weighted_loss

                loss_sums[t] += float(loss.detach())
                acc_sums[t] += float(acc)
                count[t] += 1

                log_this_step[t] = {"loss": float(loss.detach()), "acc": float(acc), "w": float(weight)}

            # Backward once with the combined loss
            self.scaler.scale(total_loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step % self.log_every == 0 or step == num_steps:
                packed = ", ".join([f"{t}: L={log_this_step[t]['loss']:.3f} A={log_this_step[t]['acc']:.3f} W={log_this_step[t]['w']:.2f}" for t in self.tasks])
                print(f"[Epoch {epoch} | Step {step}/{num_steps}] {packed}")

        # Prepare epoch averages for DWA update
        epoch_avgs = {t: (loss_sums[t] / max(count[t], 1)) for t in self.tasks}
        train_log = {
            t: {
                "loss_avg": loss_sums[t] / max(count[t], 1),
                "acc_avg": acc_sums[t] / max(count[t], 1),
                "steps": count[t],
            }
            for t in self.tasks
        }
        return train_log, epoch_avgs

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        results = {}
        for t in self.tasks:
            loader = self.val_loaders[t]
            if loader is None:
                results[t] = None
                continue

            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = self.model(x, task=t)
                loss = self.criteria[t](logits, y)
                acc = top1_acc(logits, y)
                loss_sum += float(loss)
                acc_sum += float(acc)
                n += 1

            results[t] = {
                "loss_avg": loss_sum / max(n, 1),
                "acc_avg": acc_sum / max(n, 1),
                "steps": n,
            }
        return results
    
    def rebuild_optimizer(self, lr: float, wd: float):
        # 只拿 requires_grad=True 的参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self.scaler = torch.amp.GradScaler(device=self.device)
        print_trainable_summary(self.model, prefix="[Optimizer rebuilt] ")

def train_two_stages(
    trainer: MultiTaskTrainer,
    stage1_epochs: int = 5,
    stage2_epochs: int = 5,
    # Stage1: 只训练 encoders + task_heads
    stage1_lr: float = 2e-4,
    stage1_wd: float = 0.01,
    # Stage2: 只微调 LLM（可选：仅 LoRA/MoE）
    stage2_lr: float = 1e-5,
    stage2_wd: float = 0.01,
    finetune_llm_part: str = "experts_only",  # "experts_only" | "full_llm"
):
    model = trainer.model

    # ---------- Stage 1: 冻结 LLM，只训练 encoders + task_heads ----------
    # 冻结 base_llm + moe_model（包括 MoE/LoRA）
    if hasattr(model, "base_llm"):
        set_module_trainable(model.base_llm, False)
    if hasattr(model, "moe_model"):
        set_module_trainable(model.moe_model, False)

    # 训练 encoders + task_heads
    set_module_trainable(model.encoders, True)
    set_module_trainable(model.task_heads, True)

    # 重建优化器
    trainer.rebuild_optimizer(lr=stage1_lr, wd=stage1_wd)

    print("=== Stage 1: train encoders + task_heads; freeze LLM ===")
    # 进行 stage1_epochs 轮
    orig_max_epochs = trainer.max_epochs
    trainer.max_epochs = stage1_epochs
    trainer.train()
    trainer.max_epochs = orig_max_epochs  # 复位

    # ---------- Stage 2: 冻结 encoders + task_heads，只微调 LLM ----------
    set_module_trainable(model.encoders, False)
    set_module_trainable(model.task_heads, False)

    # 选择微调范围
    if finetune_llm_part == "experts_only":
        # 默认：只开放 LoRA/MoE 专家（或适配器）参数；其余 LLM 权重保持冻结
        set_module_trainable(model.base_llm, False)
        set_module_trainable(model.moe_model, False)
        # 精准打开 experts
        for n, p in model.moe_model.named_parameters():
            if "experts" in n or "lora" in n or "gate" in n:
                p.requires_grad = True
    elif finetune_llm_part == "full_llm":

        set_module_trainable(model.base_llm, True)
        set_module_trainable(model.moe_model, True)
    else:
        raise ValueError("finetune_llm_part must be 'experts_only' or 'full_llm'.")

    # 重建优化器 —— 通常更小学习率
    trainer.rebuild_optimizer(lr=stage2_lr, wd=stage2_wd)

    print(f"=== Stage 2: finetune LLM ({finetune_llm_part}); freeze encoders + heads ===")
    trainer.max_epochs = stage2_epochs
    trainer.train()
    trainer.max_epochs = orig_max_epochs

# -----------------------
# Datasets
# -----------------------
def load_and_split_dataset(pt_path, batch_size, task, train_ratio=0.8, seed=42):
    torch.manual_seed(seed)
    data = torch.load(pt_path, weights_only=False)
    x, y = data['data'], data['label']
    dataset = MyDataset(x, y, task)
    total = len(dataset)
    train_len = int(total * train_ratio)
    test_len = total - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader


def main(configs):
    # ----------- configs (match your model init) -----------

    device = f"cuda:{configs.gpu}" if torch.cuda.is_available() else "cpu"

    # ----------- initialize model -----------
    print("Initializing model ...")
    
    model = LLM4RIM(configs).to(device)
    
    # ----------- build real datasets -----------
    occ_train, occ_val = load_and_split_dataset(
        pt_path='./datasets/occupancy.pt',
        batch_size= configs.batch_size,
        task='occupancy',
        train_ratio= 0.8
    )
    br_train, br_val = load_and_split_dataset(
        pt_path='./datasets/breathing.pt',
        batch_size= configs.batch_size,
        task='breathing',
        train_ratio=0.8
    )
    ge_train,ge_val = load_and_split_dataset(
        pt_path='./datasets/gesture.pt',
        batch_size=configs.batch_size,
        task='gesture',
        train_ratio=0.8
    )
    dr_train, dr_val = load_and_split_dataset(
        pt_path='./datasets/drowsiness.pt',
        batch_size=configs.batch_size,
        task='drowsiness',
        train_ratio=0.8
    )
    

    # ----------- pack task specs -----------
    task_specs = [
        TaskSpec(name="occupancy", train_ds=occ_train, val_ds=occ_val, batch_size=8),
        TaskSpec(name="breathing", train_ds=br_train, val_ds=br_val, batch_size=8),
        TaskSpec(name="gesture", train_ds=ge_train, val_ds=ge_val, batch_size=4),
        TaskSpec(name="drowsiness", train_ds=dr_train, val_ds=dr_val, batch_size=4),
    ]

    # ----------- trainer -----------
    trainer = MultiTaskTrainer(
        model=model,
        task_specs=task_specs,
        lr=configs.lr,
        wd=0.01,
        max_epochs=configs.epochs,
        ckpt_dir="./checkpoints_llm4rim",
        grad_clip=1.0,
        log_every=10,
        device=device,
    )
    # trainer.train()
    train_two_stages(
        trainer,
        stage1_epochs=int(configs.epochs),
        stage2_epochs=configs.epochs-int(configs.epochs),
        stage1_lr=configs.lr,
        stage1_wd=0.01,
        stage2_lr=configs.lr,
        stage2_wd=0.01,
        finetune_llm_part="experts_only",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_mode',type=int, default=1, help='usage of model, options:[0:train, 1:test], default:1')
    parser.add_argument('--llm_name', type=str, required=True,help='name of llm')
    parser.add_argument('--token_len', type=int, default=256,help='token length')
    # MoE Settings
    parser.add_argument('--experts_rank', type=int, default=8, help='lora rank')
    parser.add_argument('--experts_scale',type=float, default=2.0, help='lora alpha')
    parser.add_argument('--experts_num', type=int, default=8, help='the number of experts')
    parser.add_argument('--experts_num_per_tok', type=int, default=2, help='top k')
    
    # Training Settings
    parser.add_argument('--batch_size', type=int, default=32,help='batch size')
    parser.add_argument('--epochs', type=int, default=50,help='epochs')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--gpu', type=int, default=0,help='gpu id')
    
    args = parser.parse_args()

    main(args)
