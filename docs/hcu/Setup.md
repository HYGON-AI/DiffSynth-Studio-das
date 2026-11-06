# HCU setup

Use the following pinned DTK/HCU tuple. Install the wheels from the internal
package source maintained for the target container; do not mix versions from
another DTK release.

| Package | Validated version |
|---|---|
| torch | `2.10.0+das.opt1.dtk2604.2608131652.g28498c` |
| DeepSpeed | `0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df` |
| FlashAttention | `2.8.4+dtk2604.torch2100.2609190008.g29b89d` |
| PEFT | `0.21.2` |
| Accelerate | `1.15.0` |
| transformers | `5.20.2` |
| lightop | `0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3` |
| AITER | `0.1.6+dtk2604.torch2100.2609161529.ge4d834` |

With the exact wheels already obtained for this DTK image, install them from
the local wheel directory (substitute its absolute path):

```bash
python -m pip install --no-index --find-links /path/to/hcu-wheels \
  'torch==2.10.0+das.opt1.dtk2604.2608131652.g28498c' \
  'deepspeed==0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df' \
  'flash-attn==2.8.4+dtk2604.torch2100.2609190008.g29b89d' \
  'peft==0.21.2' 'accelerate==1.15.0' 'transformers==5.20.2' \
  'lightop==0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3' \
  'aiter==0.1.6+dtk2604.torch2100.2609161529.ge4d834'
python -m pip install --no-deps -e .
```

The source adapter observed an FA import-time dependency on `pytest`; keep
that dependency in the target environment. Do not hide GPUs for import
preflight, because that FA build queries devices on import. Confirm the actual
FA2 route with `python tests/hcu/validate_kernels.py --fa`, not just a successful
import.

Before a distributed run, build the fused CPUAdam extension once on one
process:

```bash
MAX_JOBS=1 python -c "from diffsynth_hcu.cpu_adam import load_fused_cpu_adam; load_fused_cpu_adam()"
```

The build needs a C++ compiler, OpenMP, and ninja. The loader checks the
DeepSpeed 0.18.4 interface and reports a version or builder mismatch directly.
Run the H3 launcher from the repository root so `diffsynth_hcu` and the C++
source package are importable.
