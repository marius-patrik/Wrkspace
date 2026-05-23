# MemoryConsolidator: Continual-Learning Memory Scaffold

This note documents a proposed `MemoryConsolidator` module for PyTorch and Avalanche projects. The design targets continual learning and reinforcement learning workflows where the model needs a practical mixture of exemplar replay, prioritized sampling, generative replay, and consolidation penalties.

## Purpose

`MemoryConsolidator` is intended to be a reusable plugin-style component that can be dropped into supervised continual-learning experiments or online RL training loops. It provides:

- a prioritized replay buffer for real experiences,
- VAE-based generative replay for offline/sleep-phase sampling,
- EWC regularization for protecting important weights,
- Avalanche-compatible lifecycle hooks,
- simple adapters for supervised datasets and RL transitions.

The working assumption is that real replay, synthetic replay, and weight-importance regularization are complementary. Real replay preserves high-fidelity samples, generative replay adds scalable rehearsal, and EWC discourages destructive movement in parameters that were important for earlier tasks.

## Recommended defaults

```python
buffer_size = 200_000
replay_batch = 256
sleep_freq = 1_000
offline_epochs = 5
per_alpha = 0.6
per_beta_start = 0.4
per_beta_end = 1.0
vae_latent = 64
ewc_lambda = 1e3
```

## Proposed package layout

```text
memory_consolidator/
  __init__.py
  hooks.py
  buffers/
    __init__.py
    segment_tree.py
    per_buffer.py
  modules/
    __init__.py
    vae.py
  plugins/
    __init__.py
    ewc.py
  adapters/
    __init__.py
    supervised.py
    rl.py
  tests/
    test_per_buffer.py
    test_adapters.py
    test_plugin_integration.py
  pyproject.toml
  README.md
```

## Public API

```python
from memory_consolidator import MemoryConsolidator, register_avalanche_plugin

mc = MemoryConsolidator(
    buffer_size=200_000,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_end=1.0,
    per_beta_steps=200_000,
    replay_batch=256,
    sleep_freq=1000,
    offline_epochs=5,
    vae_latent=64,
    ewc_lambda=1e3,
    device="cuda",
)

mc.store(ex)
batch = mc.sample(batch_size=256, mode="online")
batch = mc.sample(batch_size=256, mode="offline")
mc.sleep_pass(epochs=5, mode="hybrid")
```

The intended lifecycle is:

1. Convert incoming data to normalized experience dictionaries.
2. Store experiences in the prioritized replay buffer.
3. Sample online batches from real experiences.
4. Periodically run a sleep pass to refresh replay behavior and train the generator.
5. Use EWC after task boundaries to reduce forgetting.

## Core implementation sketch

```python
from .hooks import register_avalanche_plugin
from .buffers.per_buffer import PrioritizedReplayBuffer
from .modules.vae import VAE
from .plugins.ewc import EWCPlugin


class MemoryConsolidator:
    def __init__(self, *, buffer_size=200_000, replay_batch=256,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0, per_beta_steps=200_000,
                 sleep_freq=1000, offline_epochs=5, vae_latent=64, ewc_lambda=1e3,
                 device="cuda"):
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_steps=per_beta_steps,
        )
        self.replay_batch = replay_batch
        self.sleep_freq = sleep_freq
        self.offline_epochs = offline_epochs
        self.device = device
        self.vae = VAE(latent_dim=vae_latent).to(device)
        self.ewc = EWCPlugin(lambda_=ewc_lambda)
        self._steps = 0

    def store(self, ex):
        self.buffer.add(ex)

    def sample(self, batch_size, mode="online"):
        if mode == "online":
            return self.buffer.sample(batch_size)
        if mode == "offline":
            return self._sample_offline(batch_size)
        raise ValueError("mode must be 'online' or 'offline'")

    def sleep_pass(self, epochs, mode="hybrid"):
        if mode in ("generative", "hybrid"):
            self._train_vae_from_buffer(epochs)
        if mode in ("exemplar", "hybrid"):
            self.buffer.refresh_priorities()

    def on_batch_end(self, *args, **kwargs):
        self._steps += 1
        if self._steps % self.sleep_freq == 0:
            self.sleep_pass(self.offline_epochs, mode="hybrid")

    def on_epoch_end(self, *args, **kwargs):
        self.ewc.on_epoch_end(*args, **kwargs)
```

## Prioritized replay buffer

The replay buffer uses proportional prioritization with a sum tree. Priorities should be updated from TD error in RL, supervised loss in classification, or another domain-specific surprise signal.

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=200_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.pos = 0
        self.data = []
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0
        self.step = 0

    def add(self, ex):
        if len(self.data) < self.capacity:
            self.data.append(ex)
        else:
            self.data[self.pos] = ex
        p = self.max_priority ** self.alpha
        self.priorities.update(self.pos, p)
        self.pos = (self.pos + 1) % self.capacity

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.max_priority = max(self.max_priority, float(p))
            self.priorities.update(i, float(p) ** self.alpha)
```

## VAE generative replay

The VAE is used during sleep passes to learn an observation generator from stored examples. Offline samples are decoded from latent noise. In supervised settings, this should eventually be extended to conditional generation so labels or task IDs can be generated coherently with observations.

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=64, in_ch=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.enc_lin = nn.Linear(128 * 8 * 8, 2 * latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid(),
        )
```

## EWC regularization

EWC stores a parameter snapshot and Fisher estimate after task training. During later training, it adds a quadratic penalty for moving important parameters away from their stored values.

```python
penalty = ewc_lambda * sum(F_i * (theta_i - theta_star_i) ** 2)
```

The provided scaffold should expose `compute_fisher(model, dataloader, device, n_batches)` so improved estimators can be swapped in later. Fisher computation has non-trivial runtime cost, so it should be scheduled only at task boundaries or deliberate consolidation points.

## Avalanche hooks

```python
def register_avalanche_plugin(strategy, mc):
    class _MCPlugin:
        def before_backward(self, strategy, **kwargs):
            if hasattr(strategy.model, "ewc_plugin") and strategy.mb_y is not None:
                strategy.loss += strategy.model.ewc_plugin.penalty(strategy.model)

        def after_training_iteration(self, strategy, **kwargs):
            mc.on_batch_end()

        def after_training_epoch(self, strategy, **kwargs):
            mc.on_epoch_end()

    strategy.plugins.append(_MCPlugin())
    if hasattr(mc, "ewc") and hasattr(strategy, "model"):
        strategy.model.ewc_plugin = mc.ewc
    return strategy
```

## Dataset adapters

Supervised examples should use a compact dictionary format:

```python
def to_supervised_experience(x, y, task_id=None):
    if x.dtype != torch.uint8:
        x = (x * 255).clamp(0, 255).byte()
    return {"obs": x, "y": int(y), "task": task_id}
```

RL transitions should follow the same pattern:

```python
def to_rl_experience(obs, act, rew, next_obs, done, task_id=None):
    return {
        "obs": obs,
        "act": act,
        "rew": float(rew),
        "next_obs": next_obs,
        "done": bool(done),
        "task": task_id,
    }
```

## Example: Split-MNIST with Avalanche

```python
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.strategies import Naive
import torch
import torch.nn as nn
import torch.optim as optim
from memory_consolidator import MemoryConsolidator, register_avalanche_plugin

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
strategy = Naive(model, optimizer, torch.nn.CrossEntropyLoss(), train_mb_size=128)

mc = MemoryConsolidator(device="cuda" if torch.cuda.is_available() else "cpu")
strategy = register_avalanche_plugin(strategy, mc)

benchmark = SplitMNIST(n_experiences=5, fixed_class_order=[0,1,2,3,4,5,6,7,8,9])
for exp in benchmark.train_stream:
    strategy.train(exp)
    # Optional after each task:
    # mc.ewc.compute_fisher(model, exp.dataloader)
```

## Tests to add

Minimum tests:

- `test_per_buffer.py`: verifies sampling, priority updates, and importance weights.
- `test_adapters.py`: verifies supervised and RL dict schemas.
- `test_plugin_integration.py`: verifies Avalanche hook registration and sleep-pass cadence.

Useful acceptance checks:

- buffer capacity is respected,
- priorities are monotonic with supplied surprise scores,
- offline samples match expected tensor shape and dtype,
- EWC penalty is zero before Fisher initialization,
- EWC penalty is positive after parameter movement.

## Limitations

Generative replay can introduce sample bias, especially if the VAE underfits or collapses on older tasks. EWC adds Fisher-estimation overhead and may misestimate parameter importance when the data stream is small or non-stationary. PER also depends heavily on meaningful priority updates; stale or noisy priorities can over-focus training on unhelpful examples.

## Primary metric

Track compute-per-retained-accuracy:

```text
GPU_hours / delta_retained_accuracy
```

Lower is better. Log this alongside wall-clock time, memory footprint, final task accuracy, average accuracy, and backward-transfer/forgetting metrics.

## Next implementation pass

1. Turn this design into a pip-installable `memory_consolidator` package.
2. Add conditional VAE support for labels and task IDs.
3. Add TD-error priority refresh for RL agents.
4. Add benchmark scripts for Split-MNIST, CIFAR class-incremental learning, and at least one Gymnasium environment.
5. Add experiment logging for retained accuracy, compute cost, replay composition, and sleep-pass frequency.
