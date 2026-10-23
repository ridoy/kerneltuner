# pip install pyobjc-framework-Metal numpy
import Metal
import time
import numpy as np

dev = Metal.MTLCreateSystemDefaultDevice()
queue = dev.newCommandQueue()

'''
TODOs
Timing instrumentation
Output theoretical flops?
Output graphs per working set size
Interactive sort of system. Or just use jupyter lol
'''

def run(src, fn, bufs, n):
    """Compile `src`, run kernel `fn` over `n` threads. `bufs` are numpy arrays bound to
    [[buffer(0)]], [[buffer(1)]], ... and are updated in place with the GPU results."""
    options = Metal.MTLCompileOptions().init()
    options.setMathMode_(2)
    lib, err = dev.newLibraryWithSource_options_error_(src, options, None)
    if lib is None: raise RuntimeError(err)
    pso, err = dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_(fn), None)
    if pso is None: raise RuntimeError(err)
    mbufs = [dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared) for a in bufs]
    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(mbufs): enc.setBuffer_offset_atIndex_(b, 0, i)
    w = min(pso.maxTotalThreadsPerThreadgroup(), n)
#    sample_buffer = dev.makeCounterSampleBuffer().alloc().init()
    enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(w, 1, 1))
    enc.endEncoding()
    start = time.time_ns()
    cmd.commit()
    cmd.waitUntilCompleted()
    end = time.time_ns()
    for a, b in zip(bufs, mbufs):
        a[...] = np.frombuffer(b.contents().as_buffer(a.nbytes), dtype=a.dtype).reshape(a.shape)
    return (end, start)


if __name__ == "__main__":
    src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void add(device const float *a, device const float *b, device float *out, uint i [[thread_position_in_grid]]) {
        out[i] = sqrt(a[i]) * sqrt(b[i]);
    }
    """
    size = 1000000000
    prev = None
    for i in range(10):
        a, b, out = np.arange(1000, dtype=np.float32), np.arange(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)
        (end, start) = run(src, "add", [a, b, out], size)
        print(f"Run {i}: {end - start} ns")
        if prev is not None:
            matches = out == prev
            print(np.any(matches == False))
        prev = out
