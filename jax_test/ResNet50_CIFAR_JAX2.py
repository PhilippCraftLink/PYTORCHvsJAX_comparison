#!/usr/bin/env python3
import os
import time
import queue
import threading
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

# --- ENVIRONMENT & PERFORMANCE TUNING ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# --- WandB Import ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


# 1) Konfiguration

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 20
NUM_CLASSES = 10
LOG_INTERVAL = 50
WANDB_PROJECT = "cifar10-resnet50_FINAL"
USE_WANDB = True

# 2) Dataset Pipeline

def create_dataset(batch_size, split="train", cache=True):
    tf.config.set_visible_devices([], "GPU") 
    
    # Load 
    ds = tfds.load("cifar10", split=split, shuffle_files=True, as_supervised=False)
    
    # Caching
    if cache:
        ds = ds.cache()

    # Augmentation & Normalization
    def augment(example):
        image = tf.cast(example["image"], tf.float32)
        image = image / 255.0
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        image = (image - mean) / std
        label = tf.cast(example["label"], tf.int32)
        return {"image": image, "label": label}

    if split == "train":
        ds = ds.shuffle(10000)
        ds = ds.repeat() 

    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds.as_numpy_iterator()

# ----------------------
#  Device Prefetcher
class DevicePrefetcher:
    def __init__(self, iterator, buffer_size=3):
        self.iterator = iterator
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                batch = next(self.iterator)
                device_batch = {
                    "image": jax.device_put(batch["image"]),
                    "label": jax.device_put(batch["label"])
                }
                self.queue.put(device_batch)
            except StopIteration:
                break

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()
    
    def stop(self):
        self.stop_event.set()


# Model (NNX)
class Bottleneck(nnx.Module):
    def __init__(self, in_features, filters, stride=1, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_features, filters, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(filters, rngs=rngs)
        self.conv2 = nnx.Conv(filters, filters, kernel_size=(3, 3), strides=(stride, stride),
                              padding=(1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(filters, rngs=rngs)
        self.conv3 = nnx.Conv(filters, filters * 4, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(filters * 4, rngs=rngs)

        if stride != 1 or in_features != filters * 4:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_features, filters * 4, kernel_size=(1, 1), strides=(stride, stride),
                         use_bias=False, rngs=rngs),
                nnx.BatchNorm(filters * 4, rngs=rngs)
            )
        else:
            self.shortcut = None

    def __call__(self, x):
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        return nnx.relu(out)

class ResNet50CIFAR(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.layer1 = self._make_layer(64, 64, 3, stride=1, rngs=rngs)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, rngs=rngs)
        self.layer3 = self._make_layer(512, 256, 6, stride=2, rngs=rngs)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2, rngs=rngs)
        self.linear = nnx.Linear(2048, NUM_CLASSES, rngs=rngs)

    def _make_layer(self, in_f, out_f, blocks, stride, rngs):
        layers = []
        layers.append(Bottleneck(in_f, out_f, stride, rngs))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_f * 4, out_f, 1, rngs))
        return nnx.Sequential(*layers)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.linear(x)

# Training Step
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(model, grads)  
    # Optional für Stats
    grad_norm = optax.global_norm(grads)
    return loss, grad_norm

# 5) Main Run
if __name__ == "__main__":
    print(f"Devices: {jax.devices()}")
    
    if USE_WANDB and _WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_PROJECT,
            config={"batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "epochs": EPOCHS}
        )

    # Init Model & Optimizer
    rngs = nnx.Rngs(42)
    model = ResNet50CIFAR(rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    # Dataset Setup
    print("Initialisiere Dataset Pipeline...")
    ds_numpy_iter = create_dataset(BATCH_SIZE, split="train", cache=True)
    
    prefetcher = DevicePrefetcher(ds_numpy_iter, buffer_size=4)

    # JIT Warmup 
    print("Kompiliere (Warmup)...")
    warmup_batch = next(prefetcher) 
    t0 = time.time()
    loss_val, _ = train_step(model, optimizer, warmup_batch)
    loss_val.block_until_ready()
    print(f"✅ Compilation finished in {time.time() - t0:.4f}s")

    # --- Training Loop ---
    STEPS_PER_EPOCH = 50000 // BATCH_SIZE
    
    global_step = 0
    
    print("\nStarte Training...")
    
    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            
            # Tracking Variablen für die Epoche
            acc_loss = 0.0
            last_log_time = time.time()
            
            for step in range(STEPS_PER_EPOCH):
                batch = next(prefetcher) 
                
                loss, grad_norm = train_step(model, optimizer, batch)
                
                # Logging Interval
                if global_step % LOG_INTERVAL == 0:
    
                    current_loss = loss.item()
                    current_gn = grad_norm.item()
                    acc_loss += current_loss
                    
                    # Performance Messung
                    now = time.time()
                    imgs_per_sec = (BATCH_SIZE * LOG_INTERVAL) / (now - last_log_time)
                    last_log_time = now
                    
                    # Memory Stats
                    mem_stats = jax.local_devices()[0].memory_stats() or {}
                    
                    if USE_WANDB and _WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": current_loss,
                            "train/grad_norm": current_gn,
                            "perf/imgs_per_sec": imgs_per_sec,
                            "gpu/mem_allocated_mb": mem_stats.get("bytes_in_use", 0) / 1024**2,
                            "gpu/mem_peak_mb": mem_stats.get("peak_bytes_in_use", 0) / 1024**2,
                            "epoch": epoch + 1
                        }, step=global_step)
                        
                    print(f"Ep {epoch+1} | Step {step}/{STEPS_PER_EPOCH} | Loss: {current_loss:.4f} | {imgs_per_sec:.1f} img/s")

                global_step += 1

            epoch_dur = time.time() - epoch_start
            print(f" >> Epoch {epoch+1} fertig in {epoch_dur:.2f}s")
            
    except KeyboardInterrupt:
        print("Training unterbrochen.")
    finally:
        prefetcher.stop()
        if USE_WANDB and _WANDB_AVAILABLE:
            wandb.finish()