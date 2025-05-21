using LinearAlgebra, GPUArrays
using CUDA
using BenchmarkTools

gpu = CuArray # CUDA Array type
# gpu = MtlArray # Metal Array type

# CPU Arrays
N = 1000
A = randn(Float32, N, N)
B = randn(Float32, N, N)
C = similar(A, (N, N))
print("CPU")
mul!(C, A, B)
@btime mul!(C, A, B) # C = A B

# Move data to GPUArray format
# Moves types from Matrix to a GPU compatible type
# (Requires multiple dispatch to support GPU Matrix types!)
# mul! does support GPU arrays
Agpu = A |> gpu
Bgpu = B |> gpu
Cgpu = similar(Agpu, (N, N))
print("GPU")
mul!(Cgpu, Agpu, Bgpu)
@btime begin
    mul!(Cgpu, Agpu, Bgpu) # C = A B

    # Blocker to wait for GPU calculation to finish
    CUDA.synchronize()
end


# Blank function which is compatible with both CPU and GPU Arrays
function foo(A::AbstractArray)
end


# Blank function which is compatible with only GPU Arrays, from any provider
function foo(A::AbstractGPUArray)
end


# Blank function which is compatible with only CUDA Arrays
function foo(A::CuArray)
end

b = randn(Float32, N)
b_gpu = b |> gpu


# Power of GPU compute dependant on architecture - missing functionality in some cases
# Metal (Apple GPU) missing eigen() natively as an example
# Means adapting code to be GPU compatible may need some thought

# Attempt to solve linear system on the GPU
Agpu \ b_gpu


# LU factorisation
luF = lu(Agpu)

# Convert back to CPU array
A_2 = Array(Agpu) # Equivalent to A_2 = Agpu |> Array