# NHITS Hyperparameter Configuration Explained

This configuration defines the **hyperparameter search space** for a forecasting model using the NHITS architecture, typically with:

- Ray Tune for hyperparameter optimization
- NeuralForecast for time series forecasting

`tune.choice(...)` means:

> “During hyperparameter tuning, try one of these values.”

Ray Tune will test multiple parameter combinations to find the best-performing model.

---

# Configuration

```python
config_nhits = {
    "input_size": tune.choice([20, 40, 60, 80]),
    "max_steps": tune.choice([300, 500, 700]),
    "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
    "batch_size": tune.choice([16, 32, 64]),
    "windows_batch_size": tune.choice([128, 256, 512]),

    "n_pool_kernel_size": tune.choice([[2, 2, 1], [3, 2, 1]]),
    "n_freq_downsample": tune.choice([[8, 4, 1], [4, 2, 1]]),

    "scaler_type": tune.choice(["robust", "standard"]),
    "random_seed": tune.choice([42, 123, 2026]),
}
```

---

# Parameter Explanations

## `input_size`

```python
"input_size": tune.choice([20, 40, 60, 80])
```

Number of historical timesteps used as model input.

Example:
- `input_size=60`
→ the model uses the last 60 observations to forecast the future.

### Impact

- Smaller values:
  - less historical context
  - faster training

- Larger values:
  - more temporal information
  - heavier computation

### Typical usage

- Long seasonal patterns → larger input size
- Short/noisy series → smaller input size

---

## `max_steps`

```python
"max_steps": tune.choice([300, 500, 700])
```

Maximum number of optimization/training steps.

A step corresponds to:
- one forward pass
- one backward pass
- one weight update

### Impact

- Too low → underfitting
- Too high → overfitting and slower training

---

## `learning_rate`

```python
"learning_rate": tune.choice([1e-3, 5e-4, 1e-4])
```

Controls how quickly the model updates its weights.

| Value | Behavior |
|---|---|
| `1e-3` | faster but less stable |
| `5e-4` | balanced |
| `1e-4` | slower but more stable |

---

## `batch_size`

```python
"batch_size": tune.choice([16, 32, 64])
```

Number of series/samples processed together in each training step.

### Effects

### Small batch size
- lower GPU memory usage
- noisier gradients
- sometimes better generalization

### Large batch size
- more stable training
- better GPU utilization
- requires more VRAM

---

## `windows_batch_size`

```python
"windows_batch_size": tune.choice([128, 256, 512])
```

Specific to NHITS/NBEATS models.

Defines how many temporal windows are sampled per batch.

Important distinction:

- `batch_size`
  → number of series

- `windows_batch_size`
  → number of extracted windows

### Effects

Higher values:
- more stable optimization
- better GPU utilization
- higher memory usage

This is often one of the most influential parameters.

---

# NHITS Architecture Parameters

## `n_pool_kernel_size`

```python
"n_pool_kernel_size": tune.choice([[2, 2, 1], [3, 2, 1]])
```

Pooling kernel sizes for the different NHITS stacks.

NHITS works at multiple temporal resolutions:
- coarse
- medium
- fine

This parameter controls signal compression.

Example:

```python
[2, 2, 1]
```

means:
- stack 1 → pooling factor 2
- stack 2 → pooling factor 2
- stack 3 → pooling factor 1

### Effects

Larger pooling:
- captures long-term trends
- loses local details

Smaller pooling:
- preserves fine-grained information

---

## `n_freq_downsample`

```python
"n_freq_downsample": tune.choice([[8, 4, 1], [4, 2, 1]])
```

Frequency downsampling factor for each stack.

Controls the temporal resolution used in different layers.

Example:

```python
[8, 4, 1]
```

means:
- stack 1 → heavily compressed resolution
- stack 2 → medium resolution
- stack 3 → full resolution

### Effects

Higher values:
- better global trend modeling
- fewer local details

Lower values:
- more sensitivity to local patterns

---

## `scaler_type`

```python
"scaler_type": tune.choice(["robust", "standard"])
```

Defines the data normalization method.

---

### `"standard"`

StandardScaler:

```python
z = (x - mean) / std
```

Uses:
- mean
- standard deviation

Sensitive to outliers.

---

### `"robust"`

RobustScaler:
- uses median
- uses IQR (interquartile range)

More robust to outliers.

Typically better for:
- financial series
- noisy datasets

---

## `random_seed`

```python
"random_seed": tune.choice([42, 123, 2026])
```

Random seed used for:
- weight initialization
- sampling
- shuffling

Useful for:
- reproducibility
- testing model stability

If results vary a lot across seeds:
→ the model may be unstable.

---

# What This Configuration Is Doing

This search space tells Ray Tune:

> “Try different combinations of:
- input windows
- learning rates
- batch sizes
- NHITS architectural parameters
- scaling methods
- random seeds

and find the best-performing forecasting model.”

---

# Total Search Space Size

The total number of possible combinations is:

```python
4 * 3 * 3 * 3 * 3 * 2 * 2 * 2 * 3
```

Result:

```python
7776
```

So the search space contains **7,776 possible configurations**.

In practice:
- Ray Tune usually samples only part of the space
- unless a full grid search is explicitly used