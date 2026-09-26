# Issue #168 prepared grid/XC real-device evidence

This archive records real-device qualification of the typed `device_fused`
and `host_unfused` grid/XC schedules added for issue #168. The original
`cc3fc40d` records cover the prepared fixed-density boundary. A later
`39c31ff8` follow-up adds the native CUDA KS execution seam and qualifies the
public complete energy-plus-analytic-force endpoint.

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

## Complete public KS energy-plus-force endpoint

The follow-up source revision
`39c31ff8bfd9e532aee7ac02638b4ea64089d31a` exposes the execution schedule
through the native KS options v3 suffix. A fresh CUDA build completed 224/224
targets and linked `libvibeqc.so` with SHA-256
`1b3d8c98b1bb5d86fac01007fa21e6f8d60241b03f60d826cd7326baf44a54ff`.

The qz job `vibeqc-168-endpoint-c124-39c31ff8` ran on the
`4090-cuda12.4` compute group with one NVIDIA RTX 4090, 10 CPU cores and
100 GiB memory. It retained independent compile-cold executions, then seven
interleaved synchronized warm pairs for each complete public
`properties=("energy", "forces")` workload:

| workload | host-unfused median | device-fused median | fused speedup | bootstrap lower 95% |
| --- | ---: | ---: | ---: | ---: |
| H2, batch 1 | 0.85845 s | 0.76903 s | 1.116x | 1.085x |
| water, batch 1 | 3.83532 s | 3.36196 s | 1.141x | 1.134x |
| H2, batch 2 | 1.70825 s | 1.49115 s | 1.146x | 1.140x |

All three calls to `dft_endpoint_gate` passed. The largest retained
energy/force/translation discrepancies were respectively
`1.85e-13`, `1.38e-14`, and `5.78e-15`; paired SCF iteration counts
matched exactly. Changed-geometry replay also remained matched: H2 differed by
`2.0e-15` energy and `1.67e-15` maximum force component, while water
differed by `8.53e-14` and `1.29e-14`.

Compile-cold wall times were 31.05/30.92 s for H2 and 172.63/153.38 s for
water (device-fused/host-unfused). Cold compile is therefore reported
separately and is not used to manufacture the steady-state promotion result.

Raw persistent evidence is retained at
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0168-dft09/public-endpoint-39c31ff8-c124`.
The retained hashes are:

| file | SHA-256 |
| --- | --- |
| `run.py` | `d6687c37eddccdab30907bb1548da16bee9d007de6fdbd8920181951032fff3f` |
| `run.log` | `3ef8d69f5530053dbfcee7d4951bd74e27f135a1654eae598023c1933892db27` |
| `evidence.json` | `40548b79b30474ad400ae8e145cc9611bec97cef7bc4b87dc8beec8825253b52` |

The raw runner recorded the schedule *family* hashes
`0780336c...330c4` (host-unfused) and `83ce06a9...ee1c9`
(device-fused). The executed public KS options used the default
`point_tile=256`; the corresponding resolved schedule identities are
`83f03306...53423` and `99f2b44c...3b5c`. The raw endpoint file is
therefore retained as benchmark evidence and is not presented as a directly
installable local-profile artifact. Runtime profile reuse still requires a
fully validated bundle with matching resolved schedule/source hashes.

## Reproduction boundary

The repository sources were prepared from the exact revisions above in detached
qz worktrees. The GPU jobs used the Inspire project
`原子级化学反应基座模型2.0`, workspace `可上网GPU资源`, compute group
`4090-cuda13.2-2` for the first prepared qualification and
`4090-cuda12.4` for the complete public endpoint, with quota `1,10,100`.
The implementation Agent Note records the schedule/profile invariants and
native-KS follow-up.
