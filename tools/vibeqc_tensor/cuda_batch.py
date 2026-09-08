"""Independent prepared tensor states grouped by compatible compiled plans."""

from concurrent.futures import ThreadPoolExecutor

from .cuda_execute import PreparedCuda, compile_cuda
from .types import checked_size


class PreparedTensorBatch:
    """Prepare shape buckets while charging the sum of concurrent plan peaks.

    Identical plans share only immutable compiled code. Every system owns its
    inputs, outputs, stream, cuBLAS handle and scratch. The loop is per complete
    system, never per contraction. Python executor/thread objects and previous
    caller-retained result sets remain outside numeric-buffer accounting.
    """

    def __init__(self, plans, compiler, cache, *, max_bytes, device=0):
        plans = tuple(plans)
        checked_size(max_bytes, "batch byte budget")
        if not plans:
            raise ValueError("tensor batch requires at least one plan")
        self.peak_bytes = checked_size(
            sum(p.peak_bytes for p in plans), "concurrent tensor peaks"
        )
        if self.peak_bytes > max_bytes:
            raise ValueError("infeasible tensor batch byte budget")
        self.items = []
        self.buckets = {}
        artifacts = {}
        try:
            for i, plan in enumerate(plans):
                key = plan.identity
                self.buckets.setdefault(key, []).append(i)
                if key not in artifacts:
                    artifacts[key] = compile_cuda(plan, compiler, cache)
                self.items.append(PreparedCuda(plan, artifacts[key], device=device))
        except BaseException:
            self.close()
            raise

    def execute(self, feeds, *, workers=1):
        """Return results in system order even with independent concurrent streams."""
        feeds = tuple(feeds)
        if len(feeds) != len(self.items):
            raise ValueError("tensor batch requires one feed mapping per system")
        if type(workers) is not int or not 1 <= workers <= len(self.items):
            raise ValueError("workers must be in 1..batch_size")
        if workers == 1:
            return tuple(
                item.execute(values)
                for item, values in zip(self.items, feeds, strict=True)
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(item.execute, values)
                for item, values in zip(self.items, feeds, strict=True)
            ]
            return tuple(f.result() for f in futures)

    def close(self):
        for item in self.items:
            item.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
