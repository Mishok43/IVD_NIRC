# Neural Two-Level Monte Carlo Real-Time Rendering

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70050)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Neural Incident Radiance Cache (NIRC)** for efficient fully-fused neural network-based Global Illumination integration, combined with **Two-Level Monte Carlo** for unbiased results - achieving variance reduction and performance improvement better than previous methods.

> ⭐ **Honorable Mention** from the Best Paper Award Committee at Eurographics 2025

## Authors

- [Mikhail Dereviannykh](http://mishok43.com) - Karlsruhe Institute of Technology
- Dmitrii Klepikov - Karlsruhe Institute of Technology
- [Johannes Hanika](https://jo.dreggn.org/home/) - Karlsruhe Institute of Technology
- Carsten Dachsbacher - Karlsruhe Institute of Technology

## Publication

**Neural Two-Level Monte Carlo Real-Time Rendering**  
Eurographics 2025 - Computer Graphics Forum  
[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70050) | [Project Page](https://mishok43.github.io/nirc) | [Video](https://www.youtube.com/embed/Y791SlodLOs)

## Abstract

We introduce an efficient **Two-Level Monte Carlo** (subset of Multi-Level Monte Carlo, MLMC) estimator for real-time rendering of scenes with global illumination. Using MLMC we split the shading integral into two parts: the radiance cache integral and the residual error integral that compensates for the bias of the first one. For the first part, we developed the **Neural Incident Radiance Cache (NIRC)** leveraging the power of tiny neural networks as a building block, which is trained on the fly. The cache is designed to provide a fast and reasonable approximation of the incident radiance: an evaluation takes **2-25×** less compute time than a path tracing sample. This enables us to estimate the radiance cache integral with a high number of samples and by this achieve faster convergence. For the residual error integral, we compute the difference between the NIRC predictions and the unbiased path tracing simulation. Our method makes no assumptions about the geometry, materials, or lighting of a scene and has only few intuitive hyper-parameters.

## Citation

```bibtex
@article{https://doi.org/10.1111/cgf.70050,
  author = {Dereviannykh, Mikhail and Klepikov, Dmitrii and Hanika, Johannes and Dachsbacher, Carsten},
  title = {Neural Two-Level Monte Carlo Real-Time Rendering},
  journal = {Computer Graphics Forum},
  volume = {44},
  number = {2},
  pages = {e70050},
  doi = {https://doi.org/10.1111/cgf.70050},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70050}
}
```

## ⚠️ Important Notice

**This code is provided as-is for research purposes. The codebase is in a preliminary state and may contain experimental code, debugging artifacts, and incomplete features. Use at your own discretion.**

## Requirements

- **Falcor 4.4**: This project is built on top of [Falcor 4.4](https://github.com/NVIDIAGameWorks/Falcor). Please refer to the [Falcor 4.4 documentation](https://github.com/NVIDIAGameWorks/Falcor/tree/4.4) and [Getting Started guide](https://github.com/NVIDIAGameWorks/Falcor/blob/4.4/Docs/Getting-Started.md) for installation and compilation instructions.
  - **Prerequisites for Falcor 4.4:**
    - Windows 10 version 20H2 (October 2020 Update) or newer
    - Visual Studio 2019
    - Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004
    - GPU supporting DirectX Raytracing (NVIDIA RTX series recommended)
    - NVAPI (see Falcor README for installation instructions)
- CUDA-capable GPU (tested on RTX 3080, RTX 4080)
- Windows (primary development platform)
- Visual Studio 2019 or later
- CMake 3.18 or later

## Project Structure

```
.
├── Falcor/                          # Falcor rendering framework
│   └── Source/
│       └── RenderPasses/
│           └── DummyNeuralNetwork/  # Main implementation
│               ├── DummyNeuralNetwork.cpp
│               ├── DummyNeuralNetwork.h
│               ├── PathTracer.rt.slang
│               ├── NNLaplacian.cpp
│               └── ...
├── tiny-cuda-nn/                    # Adapted tiny-cuda-nn library
│   └── src/
│       └── fully_fused_mlp.cu      # Modified MLP implementation
└── mgpt_flip.py                     # Main script to run experiments
```

### Key Components

- **`Falcor/Source/RenderPasses/DummyNeuralNetwork/`**: Contains the main implementation of NIRC
  - `DummyNeuralNetwork.cpp`: Main render pass implementation
  - `PathTracer.rt.slang`: Ray tracing shader code
  - `NNLaplacian.cpp`: Neural network laplacian pyramid implementation
  - Various compute shaders for data processing

- **`tiny-cuda-nn/src/fully_fused_mlp.cu`**: Adapted fully-fused MLP implementation from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). Please refer to the original repository for the base implementation.

## Usage

The project is run via Mogwai (Falcor's interactive viewer):

```bash
Mogwai.exe with scene script = mgpt_flip.py
```

For detailed usage instructions and parameter descriptions, please refer to the Falcor documentation and the inline code comments.

## License

If not stated otherwise, the code in this repository uses a **GPLv3 license**. If you require alternative licensing options, please contact the authors.

**Documentation and Papers:** Documentation, papers, and non-code content are licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0).

**Third-Party Libraries:** This project includes and depends on third-party open source libraries that may have different licenses. Please refer to their respective license files and documentation:
- **Falcor**: See `Falcor/LICENSE.md` for Falcor's license terms
- **tiny-cuda-nn**: See `tiny-cuda-nn/LICENSE.txt` for tiny-cuda-nn's license terms
- Other dependencies may have their own license terms

See the [LICENSE](LICENSE) file for full details.

## Acknowledgments

- Built on [Falcor](https://github.com/NVIDIAGameWorks/Falcor) by NVIDIA
- Uses adapted code from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- Project page template adapted from [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Note**: This repository contains research code that may not be production-ready. Some features may be incomplete or experimental. We welcome contributions and bug reports.
