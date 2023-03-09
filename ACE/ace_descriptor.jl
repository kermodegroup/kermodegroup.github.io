module ace_descriptor

using ACE1pack

function ACE(species::Vector{String}, N::Int, maxdeg::Int, rcut::Float64)
    basis = ace_basis(species = Symbol.(species), 
                       N = N, 
                       rcut = rcut, 
                       maxdeg=maxdeg)

    return basis
end

function get_ace_vecs(ace, atoms)
    descriptor = zeros((length(atoms), length(ace)))

    for i in 1:length(atoms)
        descriptor[i, :] = site_energy(ace, atoms, i)
    end
    return descriptor
end

end