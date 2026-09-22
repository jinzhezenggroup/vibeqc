"""Execute emitted static initialization with explicit host CUDA-transfer stubs."""

import ctypes as ct
import shutil
import subprocess

import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.validate_tensor_cuda_compile import _program


@pytest.fixture(scope="module")
def initializer(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    plan = plan_cuda(_program(), cuda_target_info("sm_120"))
    source = emit_cuda(plan, embed_static_data=False)
    begin = source.index('extern "C" int tensor_static_initialize(')
    end = source.index("static int tensor_run_impl(", begin)
    initializer = source[begin:end]
    prefix = r"""
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <cstddef>
using cudaStream_t = int;
constexpr int cudaMemcpyHostToDevice = 1;
static int copies, syncs, pending, fail_at;
int cudaMemcpyAsync(void*, const void*, size_t, int, int) {
    if (++copies == fail_at) return 1;
    ++pending;
    return 0;
}
int cudaStreamSynchronize(int) { ++syncs; pending = 0; return 0; }
void cuda_check(int status) { if (status) throw std::runtime_error("injected transfer failure"); }
void error_text(char* out, size_t size, const char* msg) { if (size) { std::strncpy(out,msg,size); out[size-1]=0; } }
struct Context {};
struct GraphContext : Context {
    std::mutex mutex;
    bool static_ready = false;
    unsigned char storage[4096]{};
    unsigned char* arena = storage;
    int stream = 1;
    void check_device() {}
};
"""
    wrapper = r"""
extern "C" void check_initialization(int mode, int* result) {
    GraphContext ctx;
    unsigned char payload[PAYLOAD_SIZE]{};
    char error[256]{};
    copies = syncs = pending = 0;
    fail_at = mode == 1 ? 2 : -1;
    result[0] = tensor_static_initialize(&ctx, payload, PAYLOAD_SIZE, error, sizeof(error));
    result[1] = pending;
    result[2] = syncs;
    result[3] = ctx.static_ready;
    const int previous_copies = copies;
    fail_at = -1;
    result[4] = tensor_static_initialize(&ctx, payload, PAYLOAD_SIZE, error, sizeof(error));
    result[5] = copies - previous_copies;
    result[6] = ctx.static_ready;
}
""".replace("PAYLOAD_SIZE", str(plan.static_data_bytes))
    root = tmp_path_factory.mktemp("static-initializer")
    path = root / "probe.cpp"
    path.write_text(prefix + initializer + wrapper)
    library = root / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            str(path),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    handle = ct.CDLL(str(library))
    handle.check_initialization.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
    handle.check_initialization.restype = None
    return handle


def test_static_payload_is_immutable_after_success(initializer: ct.CDLL) -> None:
    state = (ct.c_int * 7)()
    initializer.check_initialization(0, state)
    assert state[0] == 0 and state[3] == 1
    assert state[4] != 0, "a second initialization can overwrite plan constants"
    assert state[5] == 0 and state[6] == 1


def test_partial_upload_failure_drains_before_return_and_allows_retry(
    initializer: ct.CDLL,
) -> None:
    state = (ct.c_int * 7)()
    initializer.check_initialization(1, state)
    assert state[0] != 0 and state[3] == 0
    assert state[1] == 0, (
        "failed initialization returned with a borrowed host upload pending"
    )
    assert state[2] == 1
    assert state[4] == 0 and state[5] >= 2 and state[6] == 1
