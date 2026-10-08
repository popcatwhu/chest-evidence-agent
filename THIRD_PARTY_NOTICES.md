# Third-party materials

The MIT license in this repository covers the original project code. Model weights,
datasets, reference texts and dependencies retain their upstream licenses and terms.
They are not included in this repository.

- [MedRAX](https://github.com/bowang-lab/MedRAX): architecture inspiration, demonstration assets and evaluation metadata.
- [ChestAgentBench](https://huggingface.co/datasets/wanglab/chestagentbench): public diagnostic question benchmark. Source figures and patient histories are downloaded separately; follow its terms and attribution requirements.
- [Lingshu](https://huggingface.co/lingshu-medical-mllm/Lingshu-32B): main medical vision-language model. Consult the official model card and license before using or distributing weights.
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision): classification and anatomical segmentation tools and their checkpoints.
- [IAMJB CXR baseline](https://huggingface.co/IAMJB/chexpert-mimic-cxr-findings-baseline): model-generated auxiliary observations.
- [NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B): experimental comparison only; not the default reader. Weights use the NVIDIA OneWay Noncommercial license.
- [MedlinePlus](https://medlineplus.gov/), [NIH ClinicalInfo](https://clinicalinfo.hiv.gov/) and linked professional references: retrieval sources. Source URLs and source types are retained by the preparation scripts.

Public evaluation summaries contain aggregate measurements and limited public-case
label comparisons. They do not provide clinical validation. The repository does not
redistribute clinical images, the local case database, uploaded files or model weights.
