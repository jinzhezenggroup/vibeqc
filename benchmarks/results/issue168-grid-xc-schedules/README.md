# Issue #168 prepared grid/XC real-device evidence

This archive records the first real-device qualification of the typed
`device_fused` and `host_unfused` prepared grid/XC schedules added for
issue #168. It is deliberately **not** a complete-SCF promotion record: the
public CUDA KS/force endpoint still does not select these prepared schedules,
so fixed-density energy/potential timing cannot satisfy the required complete
energy-plus-analytic-force gate.

## Environment and source identity

- Source revision: `cc3fc40d548f082cbd67aab815824b0bd959c89b`
- qz/Inspire job: `vibeqc-168-smoke-v3-cc3fc40d`
- GPU: NVIDIA GeForce RTX 4090, 49,140 MiB
- Driver: 595.71.05
- CUDA/NVCC: 12.9.86
- Python: 3.11.16
- CMake: 3.31.10
- Native CUDA build: 261/261 targets completed
- `libvibeqc.so` SHA-256:
  `698aa44aee146b3d8dbe86e15c6734cecb8eb184d180c16d475a1ca9e235b571`

Raw persistent smoke evidence is retained at
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0168-dft09/rtx4090-smoke-v3-cc3fc40d`.
The retained hashes are:

| file | SHA-256 |
| --- | --- |
| `environment.log` | `f6c2c2e9e6fe067013ca279cd35d7fb2713355d33d8f24cf539a59c112827f5d` |
| `configure.log` | `f0254a08ee3d624d1fa54064901ed2070904dd7ab4df89cc38ae7f32d74d4f9e` |
| `build.log` | `ddfaa4ec2d9f1820229ef44e630f45429d8b9ca684ce88c5cf15aeb79d9f03b3` |
| `pytest.log` | `94e84bfb0753bcb043c9a393a137ac227c8ae1e7688a7a8e83c16099fda76470` |

The real-GPU focused suite passed **7/7 tests in 10.57 s**. It includes the
same-PBE-workload fused/unfused execution check against the independently
stored energy and potential fixture, all three prepared orbital-capacity
fallback cases, explicit failed-upload propagation, and Cartesian/spherical
local-mask replay.

## Fixed-density schedule ablation

The separate job `vibeqc-168-ablation-cc3fc40d` reused the exact library
above. For each fixture it ran one first-after-prepare execution per schedule,
then seven warm pairs with alternating fused/unfused order. Every execution
was checked against the unchanged stored PBE energy and potential reference.

| fixture | AO | points | fused warm median | unfused warm median | fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 | 2 | 32 | 9.60 ms | 17.68 ms | 1.84x |
| water | 7 | 48 | 14.60 ms | 17.95 ms | 1.23x |
| spherical-f | 16 | 32 | 11.48 ms | 18.09 ms | 1.58x |

The run retained 48 total executions. Its raw JSON and log are at
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0168-dft09/rtx4090-ablation-cc3fc40d`:

- `ablation.json` SHA-256:
  `e45f191e9ca0885da1c8ffc74b7f164b11d732344a54bb94308ff7a4e7d9637d`
- `ablation.log` SHA-256:
  `accbc2d1a85fd952c44061aa25f97072e66696c4e046f96d086dfff97b273c39`

The JSON sets `promotion_eligible=false`. These results show that both typed
lowerings execute on real hardware and quantify the prepared-boundary benefit
of keeping PBE XC/Vxc device-fused. They do not establish cold/warm/changed-
geometry complete SCF energy-plus-force promotion, batch throughput, or an
official hardware profile winner.

## Reproduction boundary

The repository source was prepared from the exact revision above in a detached
qz worktree. The GPU job used the Inspire project
`原子级化学反应基座模型2.0`, workspace `可上网GPU资源`, compute group
`4090-cuda13.2-2`, and quota `1,10,100`. The implementation Agent Note
records the schedule/profile invariants and the remaining public-KS seam.
