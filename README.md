# Raspberry Pi 5 - TensorFlow Installation Guide

This guide explains how to install TensorFlow on a Raspberry Pi 5.

## 1) Recommended environment

- Device: Raspberry Pi 5
- OS: Raspberry Pi OS 64-bit (Bookworm recommended)
- Python: 3.11 or newer
- Storage: at least 8 GB free
- Memory: 8 GB model recommended (4 GB works, but slower)

Check your environment:

```bash
uname -m
python3 --version
```

`uname -m` should show `aarch64` (64-bit ARM).

## 2) System update

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

After reboot:

```bash
sudo apt install -y python3-venv python3-pip libatlas-base-dev
```

## 3) Create and activate a virtual environment

```bash
mkdir -p ~/tf-rpi5
cd ~/tf-rpi5
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip tooling:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4) Install TensorFlow

Install the latest available version:

```bash
pip install tensorflow
```

If you need a specific version:

```bash
pip install "tensorflow==2.16.*"
```

## 5) Verify installation

Run:

```bash
python - << 'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Build with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY
```

On a typical Raspberry Pi setup, GPU acceleration is usually not available for standard TensorFlow builds.

## 6) Quick inference test

```bash
python - << 'PY'
import tensorflow as tf
import numpy as np

x = np.random.rand(1000, 10).astype(np.float32)
y = np.random.rand(1000, 1).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=2, verbose=1)
print("Training test complete.")
PY
```

## 7) Troubleshooting

### `No matching distribution found for tensorflow`

- Confirm 64-bit OS (`uname -m` => `aarch64`)
- Update pip: `python -m pip install --upgrade pip`
- Verify Python version is supported by the TensorFlow release you install

### Install is too slow or fails due to memory

- Close background apps
- Make sure swap is enabled
- Retry inside a clean virtual environment

### Need lightweight inference only

If full TensorFlow is too heavy, consider TensorFlow Lite runtime:

```bash
pip install tflite-runtime
```

## 8) Re-enter environment later

```bash
cd ~/tf-rpi5
source .venv/bin/activate
```

Deactivate when done:

```bash
deactivate
```