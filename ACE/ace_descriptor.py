def ace_descriptor(species, **kwargs):
    '''
    Generate a simple callable interface for an ACE descriptor using Julia and ACE1pack.jl

    species: list of str
        List of chemical symbols (e.g. ["Si"])

    kwargs: 
        Extra keyword args passed directly to ACE1x.ace_basis
        See https://acesuit.github.io/ACEpotentials.jl/dev/literate_tutorials/TiAl_basis/ for list of args
    '''
    from julia import Main
    Main.eval("using ASE, JuLIP, ACEpotentials")

    ace_func = Main.eval(f"basis(species; kwargs...) = ACE1x.ace_basis(elements=Symbol.(species); kwargs...)")
    ace = ace_func(species, **kwargs)

    
    ASE_atoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
    ase_to_julip_atoms = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

    py_to_julip_atoms = lambda struct: ase_to_julip_atoms(ASE_atoms(struct))

    eval_ace = Main.eval('''
        function get_ace_vecs(ace, atoms)
            descriptor = zeros((length(atoms), length(ace)))

            for i in 1:length(atoms)
                descriptor[i, :] = site_energy(ace, atoms, i)
            end
            return descriptor
        end
    
    ''')

    def calc_descriptor(atoms):
        '''
        Inner func to get descriptor vectors given an atoms object
        '''
        jl_ats = py_to_julip_atoms(atoms)

        descriptors = eval_ace(ace, jl_ats)

        return descriptors

    return calc_descriptor
