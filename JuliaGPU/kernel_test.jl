using LinearAlgebra, GPUArrays, KernelAbstractions
using CUDA
using BenchmarkTools

gpu = CuArray # CUDA Array type


### Conventional approach to writing a function
function polynomial_basis(x::AbstractVector, N::Int)
    # Construct polynomial basis up to order N
    
    # similar() means we get a result array of a type which is compatible with x
    # Means we don't need to care whether it's a CPU Array or a GPU Array
    result = similar(x, (size(x, 1), N+1))
    
    for i in 1:N+1
        # Julia Hint: ".^" means elementwise power
        result[:, i] = x.^(i-1)
    end
    return result
end

N = 90_000
Npoly = 15
A = randn(Float32, N)
Agpu = A |> gpu

println(Agpu[1, 1])

# Allow for precompilation of correct calling patterns
polynomial_basis(A, Npoly)
res = polynomial_basis(Agpu, Npoly)

res_cpu = res |> Array

print("CPU: Normal Function")
@btime begin
    polynomial_basis(A, Npoly)
end

print("GPU: Normal Function")
@btime begin
    polynomial_basis(Agpu, Npoly)
    CUDA.synchronize()
end

### KernelAbstractions kernel
@kernel function _poly_kernel!(@Const(x), result)
    # Define the polynomial basis in terms of a kernel which operates over the i loop 
    i, j = @index(Global, NTuple) # Kernel will be given an index

    result[i, j] = x[i] ^ (j-1)
    
    nothing # Tip from Christoph, as kernels aren't supposed to return anything!
end

### KernelAbstractions version of the polynomial basis
function poly_basis_ka(x::AbstractVector, N::Int)
    M = size(x, 1)
    result = similar(x, (M, N+1))

    # KernelAbstractions formalism
    # Work out what the backend is for the input x (CPU vs GPU)
    backend = KernelAbstractions.get_backend(x)

    # Construct our kernel to match the right backend
    _kernel! = _poly_kernel!(backend)

    # Apply the kernel
    _kernel!(x, result; ndrange=(M, N+1))

    return result
end

device = CPU()

# Precompilation
poly_basis_ka(A, Npoly)
KernelAbstractions.synchronize(device)

print("CPU: Kernel Function")
@btime begin
    poly_basis_ka(A, Npoly)
    KernelAbstractions.synchronize(device)
end

print("GPU: Kernel Function")
device = get_backend(Agpu)

# Precompilation
poly_basis_ka(Agpu, Npoly)
KernelAbstractions.synchronize(device)

@btime begin
    poly_basis_ka(Agpu, Npoly)
    KernelAbstractions.synchronize(device)
end

T = polynomial_basis(A, Npoly)
Tgpu = poly_basis_ka(Agpu, Npoly)

T2 = Tgpu |> Array

T ≈ T2