# ComfyUI DFT Frequency Analysis Nodes

A ComfyUI custom node pack for experimenting with Discrete Fourier Transform (DFT) decomposition of images — specifically exploring the roles of **amplitude** and **phase** in image reconstruction.

Inspired by classic DFT analysis experiments showing that phase carries the structural information of an image while amplitude encodes contrast energy distribution.

---

## Nodes

### DFT: 1/f Amplitude + Original Phase

Reconstructs an image using the **original phase** from the input image combined with a **1/f noise amplitude** (where amplitude at frequency `(fx, fy)` = `1 / sqrt(fx² + fy²)`).

This demonstrates that phase alone is sufficient to preserve most of the visible structure and contours of an image, even when the amplitude is completely replaced.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image |
| `dc_value` | FLOAT (0–1) | DC component intensity, controls average brightness of the output (default: 0.5) |

**Output:** Reconstructed image (same resolution as input)

---

### DFT: Original Amplitude + Random Phase

Reconstructs an image using the **original amplitude** combined with **randomized phase**.

This is the counterpart experiment — randomizing the phase completely destroys the image structure, even though all amplitude (energy) information is preserved. Images with strong directional frequency components (e.g., horizons, vertical tree trunks) may retain faint structural traces.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image |
| `seed` | INT | Random seed for phase generation (default: 0) |

**Output:** Reconstructed image (same resolution as input)

---

## Background

In a 2D DFT, each frequency component has two parts:

- **Amplitude (magnitude):** How strongly that frequency is present — encodes energy and texture statistics
- **Phase:** The spatial alignment of that frequency — encodes structural position and edges

Classic experiments on natural images reveal:

| Reconstruction | Amplitude | Phase | Result |
|---|---|---|---|
| Original | Original | Original | Exact input |
| Random phase | Original | Random | Structure collapses; only strong directional frequencies leave faint traces |
| 1/f amplitude | 1/f noise | Original | Structure and edges are surprisingly well preserved |

The 1/f amplitude follows the power spectrum statistics of natural images (`P(f) ∝ 1/f²`), which is why swapping in 1/f amplitude does not significantly disturb the perceptual structure carried by the phase.

---

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
└── custom_nodes/
    └── Comfyui_1fAmpIFFT/
        ├── __init__.py
        ├── nodes.py
        └── README.md
```

2. Restart ComfyUI.

3. Find the nodes under the **`image/frequency`** category in the node search.

---

## Requirements

- Python 3.8+
- `numpy` (included with ComfyUI)
- `torch` (included with ComfyUI)

No additional dependencies required.

---

## License

MIT
