using KernelAbstractions, CUDA, BenchmarkTools, GPUArrays, StaticArrays
using Atomix
using LinearAlgebra

function LJ_energy(p1, p2, eps, sigma)
    # Compute the Lennard-Jones energy E_ij corresponding to the bond E_ij

    r = norm(p1 - p2)
    x = (sigma / r)^2 # Non-dimensionalise in terms of sigma^2

    # Factor of 1/2 needed for double counting
    return 2 * eps * ((x^6 - x^3))
end

# Base function for LJ partial force component between atoms defined by p1 and p2
function LJ_force(p1, p2, eps, sigma)
    # Compute a force component dE_ij/dx_jk
    r = norm(p1 - p2)
    
    c = 24 * (2 / r^12 - 1 / r^6) / r # dE_ij/dr
    
    return (p1 - p2) .* (-c / r) # dE_ij/dr * dr/dp2
end

### KA Energy kernel for the LJ model
@kernel function _energy_kernel!(positions, eps, sigma, E)
    i = @index(Global)
    p1 = @inbounds SVector{3}(positions[i, 1], positions[i, 2], positions[i, 3])

    for j in 1:i-1
        p2 = @inbounds SVector{3}(positions[j, 1], positions[j, 2], positions[j, 3])
        @Atomix.atomic E[1] += 2 * LJ_energy(p1, p2, eps, sigma)
    end
end

### KA Force kernel
@kernel function _force_kernel1!(@Const(positions), @Const(eps), @Const(sigma), F)
    i = @index(Global)

    Fij = SVector{3}(0.0, 0.0, 0.0)

    N = size(positions, 1)

    p1 = @inbounds @views SVector{3}(positions[i, 1], positions[i, 2], positions[i, 3])

    for j in 1:N
        if i != j

            p2 = @inbounds @views SVector{3}(positions[j, 1], positions[j, 2], positions[j, 3])
            
            Fij += LJ_force(p1, p2, eps, sigma)
        end
    end
    F[i, :] .= Fij
end

function energy(positions, eps, sigma)
    N = size(positions, 1)

    E = similar(positions, (1))

    backend = KernelAbstractions.get_backend(positions)

    _kernel! = _energy_kernel!(backend)

    _kernel!(positions, eps, sigma, E; ndrange=(N, N))

    return E
end

function force(positions, sigma, eps; kernel2=false)
    N = size(positions, 1)

    F = similar(positions, (N, 3))

    backend = KernelAbstractions.get_backend(positions)

  
    _kernel! = _force_kernel1!(backend)

    _kernel!(positions, eps, sigma, F; ndrange=(N))
    
    return F
end

function LJ_analytic_forces(positions, sigma, eps)
    N = size(positions, 1)
    F = similar(positions, (N, 3))

    for i in 1:N
        for j in 1:i-1
            r = @views sqrt(sum((positions[i, :] - positions[j, :]).^2))
            c = 2 * eps * (6 * sigma^6 / r^7 - 12 * sigma^12 / r^13)
            for k in 1:3
                Fij = @views -c * (positions[i, k] - positions[j, k]) / r
                @allowscalar F[j, k] += Fij
                @allowscalar F[i, k] += -Fij
            end
        end
    end
    return F
end

N = 10_000
positions = randn(Float32, (N, 3))

gpu = CuArray

pos_gpu = positions |> gpu

eps = Float32(1.0)
sigma = Float32(1.0)

device = CPU()

#Precompilation
LJ_analytic_forces(positions, sigma, eps)
print("CPU: Analytic")
@btime begin
    LJ_analytic_forces(positions, sigma, eps)
    KernelAbstractions.synchronize(device)
end

# print("GPU: Analytic")
# device = get_backend(pos_gpu)

# # Precompilation
# LJ_analytic_forces(pos_gpu, sigma, eps)
# KernelAbstractions.synchronize(device)

# @btime begin
#     LJ_analytic_forces(pos_gpu, sigma, eps)
#     KernelAbstractions.synchronize(device)
# end


# Precompilation
force(positions, sigma, eps)
KernelAbstractions.synchronize(device)

print("CPU: Kernel Version")
device = CPU()
@btime begin
    force(positions, sigma, eps)
    KernelAbstractions.synchronize(device)
end


print("GPU: Kernel Version")
device = get_backend(pos_gpu)

# Precompilation
force(pos_gpu, sigma, eps)
KernelAbstractions.synchronize(device)

@btime begin
    force(pos_gpu, sigma, eps)
    KernelAbstractions.synchronize(device)
end
