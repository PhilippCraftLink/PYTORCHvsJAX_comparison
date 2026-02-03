import os
import time
import argparse
import random
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import FlaxViTForImageClassification, ViTConfig
import wandb
from tqdm import tqdm

# --- 1. Environment Config ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
tf.config.set_visible_devices([], 'GPU')

# --- 2. Parser Argumente ---
def get_args():
    parser = argparse.ArgumentParser(description="JAX ViT Training (Native Memory Stats)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="vit-benchmark")
    parser.add_argument("--log_freq", type=int, default=5)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --- 3. Data Pipeline ---
def preprocess_image(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label

def get_datasets(batch_size, seed):
    ds_builder = tfds.builder('imagenette/320px-v2')
    ds_builder.download_and_prepare()
    
    ds_train = ds_builder.as_dataset(split='train', shuffle_files=True)
    ds_val = ds_builder.as_dataset(split='validation', shuffle_files=False)

    num_train_examples = ds_builder.info.splits['train'].num_examples
    num_classes = ds_builder.info.features['label'].num_classes
    
    options = tf.data.Options()
    options.threading.private_threadpool_size = 16 
    
    ds_train = ds_train.with_options(options)
    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(10000, seed=seed)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, num_train_examples, num_classes

# --- 4. Model State (Helper) ---
class TrainState(train_state.TrainState):
    key: jax.Array

def create_train_state(rng, learning_rate, total_steps, num_classes):
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=num_classes,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        dropout_rate=0.0,
        attention_probs_dropout_rate=0.0,
        initializer_range=0.02,
    )
    model = FlaxViTForImageClassification(config=config, dtype=jnp.bfloat16)
    input_shape = (1, 224, 224, 3) 
    params = model.init_weights(rng, input_shape)
    
    warmup_steps = int(0.1 * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=learning_rate, warmup_steps=warmup_steps, decay_steps=total_steps, end_value=0.0
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.05)
    
    return TrainState.create(apply_fn=model.module.apply, params=params, tx=optimizer, key=rng), schedule

# --- 5. JIT ---
@jax.jit
def train_step(state, batch_images, batch_labels, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, batch_images, deterministic=False, rngs={'dropout': dropout_key}
        ).logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch_labels)
    return new_state, loss, accuracy

# --- 6. Main Loop ---
def main():
    args = get_args()
    seed_everything(args.seed)
    
    wandb.init(
        project=args.project_name, 
        name=f"JAX_JIT_BS{args.batch_size}",
        config=vars(args),
        tags=["jax", "jit", "native_mem"]
    )
    
    ds_train, ds_val, num_train, num_classes = get_datasets(args.batch_size, args.seed)
    
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    steps_per_epoch = num_train // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    
    state, lr_schedule = create_train_state(init_rng, args.lr, total_steps, num_classes)

    main_device = jax.devices()[0] 

    print("Starting training...")
    global_step = 0
    total_start_time = time.time()

    for epoch in range(args.epochs):
        train_iter = tfds.as_numpy(ds_train)
        pbar = tqdm(train_iter, total=steps_per_epoch, desc=f"Ep {epoch+1}", leave=False)
        
        step_times = []
        
        for i, batch in enumerate(pbar):
            images, labels = batch
            images = jnp.array(images, dtype=jnp.bfloat16)
            labels = jnp.array(labels, dtype=jnp.int32)
            
            rng, dropout_key = jax.random.split(rng)
            
            t0 = time.perf_counter()
            state, loss, acc = train_step(state, images, labels, dropout_key)
            jax.block_until_ready(state.params)
            t1 = time.perf_counter()
            
            step_duration = t1 - t0
            
            if global_step == 0:
                wandb.log({"system/compilation_time_seconds": step_duration, "global_step": global_step})
                print(f"\n[Info] Step 0 (Compilation) Time: {step_duration:.4f}s")
            else:
                step_times.append(step_duration)

            if global_step % args.log_freq == 0 and global_step > 0:
                avg_exec = np.mean(step_times) if step_times else 0.0
                step_times = []
                
                loss_val, acc_val = loss.item(), acc.item()
                lr_val = lr_schedule(state.step)

                # MEMORY STATS 
                mem_stats = main_device.memory_stats()
                
                # bytes_in_use      = torch.cuda.memory_allocated()
                # peak_bytes_in_use = torch.cuda.max_memory_allocated()
                
                mem_allocated = mem_stats.get('bytes_in_use', 0)
                max_mem_allocated = mem_stats.get('peak_bytes_in_use', 0)

                wandb.log({
                    "train/loss": loss_val,
                    "train/accuracy": acc_val,
                    "train/learning_rate": lr_val,
                    "system/execution_time_seconds": avg_exec,
                    # Hier sind die JAX-Äquivalente zu PyTorch:
                    "gpu/mem_allocated": mem_allocated,          
                    "gpu/max_mem_allocated": max_mem_allocated, 
                    "global_step": global_step,
                    "epoch": epoch + (i / steps_per_epoch)
                })
            
            global_step += 1

    total_time = time.time() - total_start_time
    avg_throughput = (global_step * args.batch_size) / total_time
    print(f"Training finished in {total_time:.2f} seconds.")
    wandb.log({"system/throughput_img_per_sec": avg_throughput})
    wandb.finish()

if __name__ == "__main__":
    main()