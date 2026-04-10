# LukeActsVec initial plan

Goal:
Test whether a GospelVec-style pipeline recovers stronger Luke-Acts affinity than Luke has with Matthew or Mark.

Initial Corpus:
- Matthew
- Mark
- Luke
- Acts

Method:
- preserve Qwen 3.5 9B
- preserve chunking, activation extraction, PCA denoising, and layer sweep
- minimize code changes
- document any deviation explicitly 

Current implementation status:
- Config updated to Matthew/Mark/Luke/Acts
- Data preparation updated to fetch Acts in place of John
- Steering chat updated for Acts commands
- Core extraction and vector computation left otherwise unchanged for maximal methodological fidelity

Known local limitation:
- Current laptop does not support CUDA/NVIDIA execution for QWEN 3.5 9B extraction
- Cloud GPU will be required for model execution
