# NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates
---

## Resources

- 📃 [Paper](https://arxiv.org/abs/2206.08545)
- 📚 [Project Page](https://mindslab-ai.github.io/nuwave2)
- 🎬 [Examples](https://mindslab-ai.github.io/nuwave2)
- 💻 [Code](https://github.com/mindslab-ai/nuwave2)

## Abstract

[Abstract](https://arxiv.org/pdf/2206.08545.pdf)—*
Conventionally, audio super-resolution models fixed the initial
and the target sampling rates, which necessitate the model to be
trained for each pair of sampling rates. We introduce NU-Wave
2, a diffusion model for neural audio upsampling that enables
the generation of 48 kHz audio signals from inputs of various
sampling rates with a single model. Based on the architecture of NU-Wave, NU-Wave 2 uses short-time Fourier convolution (STFC) to generate harmonics to resolve the main failure
modes of NU-Wave, and incorporates bandwidth spectral feature transform (BSFT) to condition the bandwidths of inputs
in the frequency domain. We experimentally demonstrate that
NU-Wave 2 produces high-resolution audio regardless of the
sampling rate of input while requiring fewer parameters than
other models. The official code and the audio samples are available at* https://mindslab-ai.github.io/nuwave2

## Authors

Seungu Han<sup>1,2</sup>,
Junhyeok Lee<sup>1</sup>
<br>
<sup>1</sup>*MINDsLab Inc., Republic of Korea,*<br>
<sup>2</sup>*Seoul National University, Republic of Korea*

## Citation

### Plain Text

```
Han, Seungu, and Junhyeok Lee. "NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates." arXiv preprint arXiv:2206.08545 (2022).
```

### BibTex

```
@misc{https://doi.org/10.48550/arxiv.2206.08545,
  doi = {10.48550/ARXIV.2206.08545},
  url = {https://arxiv.org/abs/2206.08545, 
  author = {Han, Seungu and Lee, Junhyeok},
  keywords = {Audio and Speech Processing (eess.AS), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences}, 
  title = {NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```