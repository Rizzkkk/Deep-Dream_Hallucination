# DeepDream: Visualizing Neural Network Perception

A deep learning project that uses gradient ascent on a pretrained **InceptionV3** model to visualize what different layers of a convolutional neural network respond to — from low-level textures to higher-level object-like hallucinations.

**Stack:** Python · TensorFlow · Keras · NumPy · Pillow · Matplotlib


---

## Overview

Convolutional neural networks are often described by what they predict, but not by what they internally *notice*. This project uses **DeepDream** to make those internal representations visible.

Instead of asking the model to classify an image, the notebook performs **gradient ascent on the input image itself**. At each step, the image is nudged in the direction that increases activations in selected layers of a pretrained InceptionV3 network. The result is a visual map of the patterns the model has learned from ImageNet.

The key idea is simple: if a layer strongly responds to certain textures, shapes, or semantic fragments, then maximizing that response reveals what the layer is biased to see.

---

## Project Goal

This notebook explores one main question:

> **What does a pretrained CNN actually learn at different depths, and how does that change the image when those features are amplified?**

To answer that, the project compares dream outputs from three layers:

| Layer | Depth | Typical effect |
|---|---|---|
| `mixed2` | Shallow | Repeated geometry, swirls, local texture |
| `mixed5` | Mid | Texture blends and partial shapes |
| `mixed8` | Deep | Eyes, fur, face-like and object-like fragments |

---

## Method

The workflow is built around a pretrained **InceptionV3** model with `include_top=False`, so only the convolutional feature extractor is used.

### 1. Feature extraction
Intermediate activations are taken from selected layers inside InceptionV3.

### 2. Gradient ascent on the image
A loss is defined as the weighted mean activation of the chosen layers. The image is then updated using gradient ascent:

- maximize `J(I)`
- update rule: `I ← I + α · ∇ᵢJ(I)`

where:
- `I` is the input image
- `J(I)` is the activation-based objective
- `α` is the step size

### 3. Two dreaming modes
The notebook includes two versions of DeepDream:

- **Single-scale DeepDream**  
  Runs gradient ascent at one fixed image resolution. Good for a quick first pass.

- **Multi-scale DeepDream (octaves)**  
  Runs the process across progressively larger resolutions and reintroduces lost detail at each stage. This usually produces richer and more detailed outputs.

### 4. Step-size sensitivity
The notebook also compares small, balanced, and overly large step sizes to show how unstable updates can degrade the output into noise.

---

## Results

The outputs follow a clear progression across network depth:

- **Shallow layers** amplify local edges, repeated curves, and texture-heavy motifs.
- **Mid-level layers** start combining those textures into more structured forms.
- **Deeper layers** produce recognizable semantic fragments such as eyes, fur, and facial patterns.

This makes the network's behavior easier to interpret: early layers behave like pattern detectors, while deeper layers reflect more abstract concepts learned from ImageNet.

---

## Repository Structure

```text
.
├── README.md
└── deepdream.ipynb
```

When the notebook is run, it also creates:

```text
deepdream_outputs/
├── original.png
├── dream_simple.png
├── dream_octaves.png
├── dream_shallow_mixed2.png
├── dream_mid_mixed5.png
├── dream_deep_mixed8.png
└── progression_frames/
```

---

## How to Run

### Option 1: Google Colab
Open the notebook in Colab using the badge above. The notebook will prompt you to upload an image. If no image is uploaded, it generates a synthetic fallback image automatically.

### Option 2: Local Jupyter environment

Install the required packages:

```bash
pip install tensorflow numpy pillow matplotlib
```

Then open the notebook:

```bash
jupyter notebook deepdream.ipynb
```

---

## What the Notebook Covers

- loading a pretrained InceptionV3 feature extractor
- preprocessing and deprocessing images
- defining an activation-maximization loss
- running one-step and iterative gradient ascent
- comparing step sizes
- comparing single-scale and multi-scale dreaming
- comparing shallow, mid, and deep layers
- exporting still images and progression frames

---

## Why This Project Matters

DeepDream is more than a visual trick. It is a simple and intuitive way to inspect what a convolutional network responds to internally.

For interpretability work, this project helps show:

- how feature hierarchy changes with depth
- why pretrained models hallucinate familiar structures
- how optimization choices such as target layer and step size affect the result
- how internal activations can be turned into something humans can inspect directly

---

## Limitations

- The model is pretrained on **ImageNet**, so the hallucinated features reflect ImageNet categories and biases.
- DeepDream is not a faithful explanation of the full decision process; it is a visualization tool, not a complete interpretability method.
- Results depend heavily on the source image, target layer, and optimization settings.

---

## Possible Next Steps

- test more layers across the InceptionV3 architecture
- compare different pretrained backbones such as ResNet or EfficientNet
- add tiled gradient ascent for larger images
- save animations directly as GIF or MP4
- compare feature amplification across different source images

