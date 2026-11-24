"""Importable CUDA workers for fl_las_main_parallel.ipynb."""

import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import multiprocessing as mp
from pathlib import Path
import random
import warnings


_DEVICE = None


def _initialize_worker(device_queue):
    import torch

    global _DEVICE
    device_id = device_queue.get()
    torch.set_num_threads(1)
    torch.cuda.set_device(device_id)
    _DEVICE = f"cuda:{device_id}"


def train_model(model, loader, client_id, epochs=1, device="cuda", *,
                checkpoint_dir, checkpoint_prefix, seed=None):
    """Use the notebook's AdamW/AMP training for both clients and the baseline."""
    import numpy as np
    import torch
    from torch.nn.functional import cross_entropy
    from torch.optim import AdamW
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
    model.to(device)
    opt = AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(loader, desc=f"{checkpoint_prefix} Epoch{ep}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # Return paths across processes, rather than tensors with shared lifetimes.
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save(state, checkpoint_dir / f"{checkpoint_prefix}_epoch{ep}.pt")

    return model.cpu().eval()


def _run_client(job):
    import timm
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(job["seed"])
    loader = DataLoader(job["dataset"], batch_size=job["batch_size"], shuffle=True,
                        num_workers=0, pin_memory=True, generator=generator)
    model = timm.create_model(job["backbone"], pretrained=False,
                              num_classes=job["num_classes"])
    model.load_state_dict(job["state_dict"], strict=True)
    train_model(model, loader, job["client_id"], epochs=job["epochs"], device=_DEVICE,
                checkpoint_dir=job["checkpoint_dir"],
                checkpoint_prefix=job["checkpoint_prefix"], seed=job["seed"])
    path = Path(job["checkpoint_dir"]) / f"{job['checkpoint_prefix']}_epoch{job['epochs']}.pt"
    return job["client_id"], str(path)


def _train_client(job):
    import torch

    try:
        return _run_client(job)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


class ParallelClientPool:
    """A persistent spawn pool; each worker stays on its assigned GPU."""

    def __init__(self, devices):
        self.devices = list(devices)
        if not self.devices:
            raise ValueError("At least one CUDA worker is required.")
        self._executor = None
        self._device_queue = None
        atexit.register(self.close)

    def _start(self):
        context = mp.get_context("spawn")
        self._device_queue = context.Queue()
        for device in self.devices:
            self._device_queue.put(device)
        self._executor = ProcessPoolExecutor(
            max_workers=len(self.devices), mp_context=context,
            initializer=_initialize_worker, initargs=(self._device_queue,))

    def train(self, jobs):
        if self._executor is None:
            self._start()
        try:
            futures = [self._executor.submit(_train_client, job) for job in jobs]
            results = [future.result() for future in as_completed(futures)]
            return [path for client_id, path in sorted(results)]
        except BaseException as error:
            self.close()
            unique_devices = list(dict.fromkeys(self.devices))
            is_oom = (type(error).__name__ == "OutOfMemoryError"
                      or "CUDA out of memory" in str(error))
            if is_oom and len(unique_devices) < len(self.devices):
                self.devices = unique_devices
                warnings.warn("CUDA memory exhausted; retrying this batch with one client per GPU.",
                              RuntimeWarning)
                return self.train(jobs)
            raise

    def close(self):
        if self._executor is not None:
            executor, self._executor = self._executor, None
            executor.shutdown(wait=True, cancel_futures=True)
            self._device_queue.close()
            self._device_queue.join_thread()
            self._device_queue = None
