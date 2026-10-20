# pip install pyobjc-framework-Metal numpy
import Metal
import numpy as np

dev = Metal.MTLCreateSystemDefaultDevice()
queue = dev.newCommandQueue()

def run(src, fn, bufs, n):
    """Compile `src`, run kernel `fn` over `n` threads. `bufs` are numpy arrays bound to
    [[buffer(0)]], [[buffer(1)]], ... and are updated in place with the GPU results."""
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    if lib is None: raise RuntimeError(err)
    pso, err = dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_(fn), None)
    if pso is None: raise RuntimeError(err)
    mbufs = [dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared) for a in bufs]
    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(mbufs): enc.setBuffer_offset_atIndex_(b, 0, i)
    w = min(pso.maxTotalThreadsPerThreadgroup(), n)
    enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(n, 1, 1), Metal.MTLSizeMake(w, 1, 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    for a, b in zip(bufs, mbufs):
        a[...] = np.frombuffer(b.contents().as_buffer(a.nbytes), dtype=a.dtype).reshape(a.shape)


if __name__ == "__main__":
    src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void add(device const float *a, device const float *b, device float *out, uint i [[thread_position_in_grid]]) {
        out[i] = a[i] * b[i];
    }
    """
    a, b, out = np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32), np.zeros(8, dtype=np.float32)
    run(src, "add", [a, b, out], 8)
    print(out)
