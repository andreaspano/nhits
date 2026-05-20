# NHITS Model — Brief Explanation

NHITS (Neural Hierarchical Interpolation for Time Series Forecasting) is a deep learning architecture designed for accurate and efficient time series forecasting.

It was introduced as an improvement over N-BEATS for:

* long forecasting horizons
* multiscale temporal patterns
* computational efficiency

---

# Core Idea

NHITS forecasts a time series by analyzing it at multiple temporal resolutions:

* coarse scale → long-term trends
* medium scale → seasonal patterns
* fine scale → local fluctuations

Instead of processing the signal at full resolution everywhere, NHITS:

1. downsamples the input
2. learns patterns at different frequencies
3. combines them hierarchically

This makes it:

* faster
* more memory efficient
* often more accurate on long horizons

---

# Architecture Overview

The model is composed of several stacks/blocks.

Each block:

1. receives a residual signal
2. extracts information at a specific temporal scale
3. produces:

   * backcast → explains past signal
   * forecast → contributes to future prediction

The residual is passed to the next block.

Conceptually:

```text
Input Series
     ↓
[Coarse Block]
     ↓
[Medium Block]
     ↓
[Fine Block]
     ↓
Final Forecast
```

---

# Key Components

## 1. Hierarchical Interpolation

NHITS predicts at lower resolutions and then interpolates back to full resolution.

Benefits:

* lower computational cost
* smoother long-horizon forecasts
* better trend extraction

---

## 2. Multi-Rate Sampling

Different stacks work at different frequencies.

Example:

* stack 1 → every 8th timestep
* stack 2 → every 4th timestep
* stack 3 → full resolution

This is controlled by:

```python
n_freq_downsample
```

---

## 3. Pooling

NHITS uses pooling to compress the signal before processing.

Controlled by:

```python
n_pool_kernel_size
```

Pooling helps the model focus on:

* global patterns
* seasonality
* trend structure

---

# Why NHITS Works Well

Compared to many transformer-based forecasting models, NHITS is:

* simpler
* faster
* less memory intensive
* strong on long forecasting horizons

It performs especially well on:

* energy forecasting
* finance
* retail demand
* sensor/IoT series

---

# Main Advantages

| Advantage                | Description                              |
| ------------------------ | ---------------------------------------- |
| Multi-scale learning     | captures trends + local patterns         |
| Efficient                | reduced computation through downsampling |
| Long-horizon forecasting | strong performance far into the future   |
| Stable training          | simpler than transformers                |
| Flexible                 | works on many time series domains        |

---

# Typical NHITS Workflow

```text
Historical Window
        ↓
Downsampling + Pooling
        ↓
MLP Blocks
        ↓
Hierarchical Forecast Components
        ↓
Interpolated Final Forecast
```

---

# Important Hyperparameters

| Parameter            | Role                          |
| -------------------- | ----------------------------- |
| `input_size`         | amount of history used        |
| `h`                  | forecast horizon              |
| `n_pool_kernel_size` | pooling/compression           |
| `n_freq_downsample`  | temporal resolution per stack |
| `batch_size`         | training batch size           |
| `learning_rate`      | optimizer update speed        |

---

# When to Use NHITS

NHITS is a strong choice when:

* forecasting horizons are long
* seasonality exists at multiple scales
* GPU memory is limited
* you want strong performance without transformers

It is often one of the best baseline deep-learning models for time series forecasting.
