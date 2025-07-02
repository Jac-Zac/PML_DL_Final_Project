# 🌀 Diffusion Models – A Comprehensive Overview

---

## 1. 📘 **DDPM: Denoising Diffusion Probabilistic Models**

- **Process**:

  - _Forward_: Gradually add Gaussian noise to an image over T timesteps
  - _Reverse_: Learn to remove noise step-by-step using a U-Net architecture

- **Loss**: Weighted MSE between predicted and actual noise at each timestep
- **Key Training Tips**:
  - Uniform timestep weights (set to 1) often **improve performance**
  - Sample timesteps from a **Gaussian distribution (centered mid-sequence)** to emphasize middle steps
  - Use cosine noise schedule for better performance than linear

---

## 2. 📗 **DDIM: Denoising Diffusion Implicit Models**

- **Process**:

  - A deterministic variant of DDPM that uses non-Markovian diffusion
  - Enables faster sampling by skipping some timesteps while maintaining sample quality
  - Interpolates between latent representations, allowing smooth latent space manipulation

- **Key Advantage**: Much faster generation with fewer steps compared to DDPM, without retraining the model

---

## 2. 🧠 **Stable Diffusion v1 – Latent Diffusion**

- **Architecture**: VAE encoder → U-Net diffusion in latent space → VAE decoder
- **Latent Space**: Uses a VAE to compress 512×512 images to 64×64×4 latent representations
- **Model**: U-Net operates on compressed latent representation (8× reduction in computation)
- **Conditioning**: Via **cross-attention** layers for text prompts using CLIP text encoder
- **Innovation**: Made high-resolution diffusion computationally feasible

---

## 3. 🔍 **Stable Diffusion v2 – Enhanced Architecture**

- **Improvements**:
  - Enhanced VAE with better reconstruction quality
  - **OpenCLIP** text encoder (instead of original CLIP) for better text understanding
  - Higher resolution training (768×768)
  - Improved noise scheduling and sampling techniques
- **Architecture**: Still uses U-Net + VAE latent diffusion approach

---

## 4. 🔄 **Stable Diffusion v3 – Transformer Revolution**

**CORRECTION**: SD3 **still uses latent space** with VAE encoding/decoding!

- **Architecture**: VAE encoder → MMDiT (Multimodal Diffusion Transformer) in latent space → VAE decoder
- **Key Innovation**: MMDiT uses separate sets of weights for image and language representations, improving text understanding and spelling
- **Flow Matching**: Replaces traditional DDPM noise prediction with rectified flow
- **Transformer**: Replaces U-Net with transformer architecture for better global modeling
- **Text Encoders**: Uses multiple text encoders (T5, CLIP) for enhanced conditioning

---

## 5. 🌊 **Flow Matching vs. DDPM**

### Traditional DDPM:

- Learns to predict noise at each timestep
- Requires many sampling steps (50-1000)
- Stochastic differential equations (SDEs)

### Flow Matching:

- Learns to match probability flow directly
- **Benefits**:
  - Faster training convergence
  - Higher fidelity outputs
  - Fewer sampling steps needed
  - More stable training dynamics
- Used in SD v3, Flux, and other modern models

---

## 6. 🧩 **U-Net vs. Transformer Architectures**

| Feature       | U-Net                                   | Transformer                             |
| ------------- | --------------------------------------- | --------------------------------------- |
| **Strengths** | Local spatial features, fast inference  | Global context modeling, better scaling |
| **Attention** | Self-attention within resolution levels | Full global attention                   |
| **Memory**    | Memory efficient                        | Higher memory requirements              |
| **Used In**   | DDPM, SD v1/v2                          | SD v3, Flux, Imagen                     |
| **Scaling**   | Limited by architecture                 | Scales better with parameters           |

---

## 7. 🧠 **Cross-Attention Conditioning**

- **Purpose**: Injects external information (text, style, etc.) into generation process
- **Mechanism**:
  - Text is encoded into embeddings
  - Image features attend to text embeddings
  - Allows fine-grained control over generation
- **Used in**: All modern conditional diffusion models

---

## 8. ⚙️ **Sampling Speed Optimization**

| Method                       | Steps Needed | Description                   | Trade-offs                    |
| ---------------------------- | ------------ | ----------------------------- | ----------------------------- |
| **Original DDPM**            | ~1000+       | Slow but highest quality      | Impractical for real-time use |
| **DDIM**                     | ~20-50       | Deterministic sampling        | Faster, slight quality loss   |
| **Progressive Distillation** | ~4-8         | Model distilled to skip steps | Requires retraining           |

---

## 12. 🔧 **Advanced Training Techniques**

### Noise Scheduling:

- **Linear**: Simple but suboptimal
- **Cosine**: Better perceptual quality
- **v-parameterization**: Improved training stability

### Loss Functions:

- **MSE**: Standard choice
- **L1**: Sometimes better for sharp details

---

## 📌 **Model Comparison Summary**

| Model        | Architecture  | Latent Space  | Key Innovation               | Best For               |
| ------------ | ------------- | ------------- | ---------------------------- | ---------------------- |
| **DDPM**     | U-Net         | No            | Original diffusion           | Research/Understanding |
| **SD v1**    | U-Net + VAE   | Yes (64×64×4) | Latent diffusion             | General use            |
| **SD v2**    | U-Net + VAE   | Yes           | Better text encoder          | Improved quality       |
| **SD v3**    | MMDiT + VAE   | Yes           | Flow matching + transformers | Text accuracy          |
| **SD 3.5**   | MMDiT-X + VAE | Yes           | QK normalization             | Stability              |
| **Flux 1.1** | Transformer   | Yes           | Hybrid architecture          | Speed + quality        |

---

### Sampling:

- Use classifier-free guidance (CFG) scale 7-15 for best results
- Try different samplers (DPM++, Euler, Heun) for speed/quality balance
- Seed control ensures reproducible results

[Stable diffusion notes](https://www.youtube.com/watch?v=n233GPgOHJg)

# Future things to review

## Predictive mean and variance via Monte Carlo approximation

Potentially this could be change we need to think about this!

### No need for MC perhaps?

```python
  mean, var = self.conv_out_la(
      feats, pred_type="nn", link_approx="mc", n_samples=100
  )
```
