# Third-Party Notices

Kokoro Server uses and distributes third-party software under its respective
licenses. The project's Apache-2.0 license does not relicense dependencies,
base-image components, NVIDIA CUDA components, or models. Package managers
retain upstream license and notice files in their installed locations.

This document identifies key components, not a complete legal bill of
materials. Consult each distribution and installed package for its complete
license terms and notices.

## Software Distributed in the Runtime Image

- [Kokoro 0.9.4](https://pypi.org/project/kokoro/0.9.4/) and
  [Misaki 0.9.4](https://pypi.org/project/misaki/0.9.4/) use Apache-2.0.
- [PyAV 13.1.0](https://pypi.org/project/av/13.1.0/) and
  [python-soundfile 0.12.1](https://pypi.org/project/soundfile/0.12.1/) use
  BSD-3-Clause.
- [spaCy `en_core_web_sm` 3.8.0](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0)
  declares MIT in its package metadata.
- [UniDic](https://pypi.org/project/unidic/) package code uses MIT/WTFPL. Its
  separately downloaded dictionary uses BSD terms.
- [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) uses GPLv3. It remains
  a separately installed system package and retains notices that its package
  supplies. The Apache-2.0 project license does not replace its GPLv3 terms.

The runtime image also contains PyTorch 2.6.0, torchaudio 2.6.0, FFmpeg,
libsndfile, Python, and other Python and system packages. Those components
retain their respective licenses and notices. The
[`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`](https://hub.docker.com/r/nvidia/cuda)
base image and its CUDA components remain subject to NVIDIA's applicable terms.

## Runtime-Downloaded Model

The image does not contain the `hexgrad/Kokoro-82M` model weights. Kokoro
Server downloads them at runtime into its configured persistent cache.

The [model card](https://huggingface.co/hexgrad/Kokoro-82M) declares the
weights under Apache-2.0. It also contains CC BY acknowledgements for training
data. Review and preserve the model card's current license, attribution, and
training-data terms when the model weights are downloaded or redistributed.
