using KernelAbstractions
using Atomix
using CUDA
using LinearAlgebra
using BenchmarkTools

gpu = CuArray # CUDA Array type

### Same Poly basis kernel as before
@kernel function _poly_kernel!(@Const(x), result)
    i, j = @index(Global, NTuple) 

    result[i, j] = x[i] ^ (j-1)
    
    nothing
end

### Expand old poly kernel
function poly_basis_ka(x::AbstractVector, N::Int)
    M = size(x, 1)
    result = similar(x, (M, N+1))

    backend = KernelAbstractions.get_backend(x)

    _kernel! = _poly_kernel!(backend)

    _kernel!(x, result; ndrange=(M, N+1))

    return result
end

### Build a forward model by evaluating the basis, then using mul! to apply weights
function poly_model_1(x::AbstractVector, weights::AbstractVector, N::Int)
    basis = poly_basis_ka(x, N)

    result = similar(x, size(x, 1))

    mul!(result, basis, weights) # result = basis * weights 

    return result
end

### Instead, try to define the full model in one kernel
### Needs a += operation to sum over basis functions
### Therefore, we need to tell the kernel to avoid race conditions
### via @Atomix.atomic
@kernel function _poly_model!(@Const(x), @Const(weights), result)

    i, j = @index(Global, NTuple)

    @Atomix.atomic result[i] += weights[j] * x[i] ^ (j-1)
    
    nothing
end

### Expand full model kernel into function
function poly_model_2(x::AbstractVector, weights::AbstractVector, N::Int)
    M = size(x, 1)
    result = similar(x, (M))

    backend = KernelAbstractions.get_backend(x)

    _kernel! = _poly_model!(backend)

    _kernel!(x, weights, result; ndrange=(M, N+1))

    return result
end

### Basis, then multiply
function method1(N::Int, Nbasis::Int)
    x = randn(Float32, N)
    xgpu = x |> gpu

    weights = randn(Float32, Nbasis + 1)
    weightsgpu = weights |> gpu

    poly_model_1(xgpu, weightsgpu, Nbasis)
    
    device = get_backend(xgpu)
    KernelAbstractions.synchronize(device)
end

### Single Call
function method2(N::Int, Nbasis::Int)
    x = randn(Float32, N)
    xgpu = x |> gpu

    weights = randn(Float32, Nbasis + 1)
    weightsgpu = weights |> gpu

    poly_model_2(xgpu, weightsgpu, Nbasis)

    device = get_backend(xgpu)
    KernelAbstractions.synchronize(device)
end