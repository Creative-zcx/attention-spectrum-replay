from __future__ import annotations

import random
from typing import Dict, List

import torch


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.storage: List[Dict[str, object]] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add_batch(self, batch: Dict[str, object]) -> None:
        bsz = int(batch["labels"].shape[0]) if torch.is_tensor(batch["labels"]) else len(batch["labels"])
        for i in range(bsz):
            item = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    item[k] = v[i].detach().cpu().clone()
                elif isinstance(v, list):
                    item[k] = v[i]
                else:
                    item[k] = v
            self.add(item)

    def add(self, item: Dict[str, object]) -> None:
        self.n_seen += 1
        if len(self.storage) < self.capacity:
            self.storage.append(item)
            return
        j = self.rng.randint(0, self.n_seen - 1)
        if j < self.capacity:
            self.storage[j] = item

    def sample(self, n: int) -> List[Dict[str, object]]:
        if not self.storage or n <= 0:
            return []
        return [self.rng.choice(self.storage) for _ in range(n)]
