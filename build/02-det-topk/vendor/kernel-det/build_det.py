# Build the deterministic persistent_topk as a standalone extension (_C_det).
# Run only on an idle box: nvcc for sm_121a, BUILD_JOBS is irrelevant (2 files).
import os, sys, torch
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
out = os.environ.get("DET_BUILD_DIR", os.path.join(here, "build"))
os.makedirs(out, exist_ok=True)
arch = os.environ.get("DET_ARCH", "121a")
mod = load(name="_C_det", sources=[os.path.join(here, "topk_det.cu"), os.path.join(here, "bindings_det.cpp")],
           extra_include_paths=[here],
           extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
           extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--expt-relaxed-constexpr", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA"],
           build_directory=out, verbose=True, is_python_module=False)
print("built:", mod)
print("op:", torch.ops._C_det.persistent_topk)
