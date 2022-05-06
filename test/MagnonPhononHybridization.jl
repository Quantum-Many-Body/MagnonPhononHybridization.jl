using MagnonPhononHybridization
using QuantumLattices
using TightBindingApproximation
using SpinWaveTheory
using QuantumLattices: wildcard
using Plots: plot, ylims!, savefig

@testset "MagnonPhononHybridization.jl" begin
    @test dmhybridization"" == Couplings(Coupling(1, ID(NID('u', wildcard), SID{wildcard}(1, wildcard))))

    term = DMHybridization(:dmp, 2.0, 1)
    @test abbr(term) == abbr(typeof(term)) == :dmp
    @test ishermitian(term) == ishermitian(typeof(term)) == true

    bond = Bond(1, Point(PID(2), [0.5, 0.5], [0.0, 0.0]), Point(PID(1), [0.0, 0.0], [0.0, 0.0]))
    operators = Operators(
        Operator(√2/2, OID(Index(PID(2), NID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(1), NID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(1), NID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(2), NID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(1), NID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(1), NID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(2), NID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(1), NID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(2), NID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(1), NID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, OID(Index(PID(2), NID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(1), NID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(2), NID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(2), NID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(2), NID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), OID(Index(PID(1), SID{1//2}(1, 'x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, OID(Index(PID(1), NID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), OID(Index(PID(2), SID{1//2}(1, 'x')), [0.5, 0.5], [0.0, 0.0]))
    )

    hilbert = Hilbert(pid=>Phonon(2)⊕Spin{1//2}(1) for pid in [bond.epoint.pid, bond.spoint.pid])
    @test expand(term, bond, hilbert) ≈ operators

    hilbert = Hilbert(pid=>Spin{1//2}(1)⊕Phonon(2) for pid in [bond.epoint.pid, bond.spoint.pid])
    @test expand(term, bond, hilbert) ≈ operators

    hilbert = Hilbert(pid=>Phonon(2)⊗Spin{1//2}(1) for pid in [bond.epoint.pid, bond.spoint.pid])
    @test expand(term, bond, hilbert) ≈ operators

    hilbert = Hilbert(pid=>Spin{1//2}(1)⊗Phonon(2) for pid in [bond.epoint.pid, bond.spoint.pid])
    @test expand(term, bond, hilbert) ≈ operators
end

@testset "plot" begin
    a = 5.773
    c = 10.057

    lattice = Lattice(Symbol("Stacked-Honeycomb"), [
            Point(PID(1), [0.0, 0.0, 0.0]), Point(PID(2), [0.0, √3/3*a, 0.0]),
            Point(PID(3), [0.0, √3/3*a, c/2]), Point(PID(4), [0.0, 0.0, c/2])
            ],
        vectors=[[1.0, 0.0, 0.0]*a, [0.5, √3/2, 0.0]*a, [0.0, 0.0, 1.0]*c],
        neighbors=Dict(1=>√(a^2/3), 2=>a, 3=>√(4a^2/3), 4=>c)
        )
    evenbond(bond) = all(point->iseven(point.pid.site), bond) ? 1 : 0
    oddbond(bond) = all(point->isodd(point.pid.site), bond) ? 1 : 0
    hilbertₘₚ = Hilbert(pid=>Phonon(3)⊕Spin{2}(1) for pid in lattice.pids)
    magneticstructure = MagneticStructure(lattice, Dict(pid=>(pid.site%4∈(1, 0) ? [0, 0, 1] : [0, 0, -1]) for pid in lattice.pids))

    Jxy₁ = SpinTerm(:Jxy₁, 1.0, 1, couplings=ising"x"+ising"y", modulate=true)
    JAxy₂ = SpinTerm(:JAxy₂, 1.0, 2, couplings=ising"x"+ising"y", amplitude=oddbond, modulate=true)
    JBxy₂ = SpinTerm(:JBxy₂, 1.0, 2, couplings=ising"x"+ising"y", amplitude=evenbond, modulate=true)
    Jxy₃ = SpinTerm(:Jxy₃, 1.0, 3, couplings=ising"x"+ising"y", modulate=true)
    Δ₁ = SpinTerm(:Δ₁, 1.0, 0, couplings=(sᶻ"")^2, amplitude=point->isodd(point.pid.site) ? 1 : 0, modulate=true)
    Δ₂ = SpinTerm(:Δ₂, 1.0, 0, couplings=(sᶻ"")^2, amplitude=point->iseven(point.pid.site) ? 1 : 0, modulate=true)
    T = PhononKinetic(:T, 0.5, modulate=true)
    V₁ = PhononPotential(:V₁, 1.0, 1, modulate=true)
    V₂₁ = PhononPotential(:V₂₁, 1.0, 2, amplitude=oddbond, modulate=true)
    V₂₂ = PhononPotential(:V₂₂, 1.0, 2, amplitude=evenbond, modulate=true)
    V₃ = PhononPotential(:V₃, 1.0, 3, modulate=true)
    V₄ = PhononPotential(:V₄, 1.0, 4, modulate=true)
    D = DMHybridization(:D, 1.0, 1, amplitude=bond->all(point->point.pid.site%4∈(1, 2), bond) ? 1 : -1, modulate=true)

    FMOAFMMP = Algorithm(:FMOAFMMP, LSWT(lattice, hilbertₘₚ, (Jxy₁, JAxy₂, JBxy₂, Jxy₃, Δ₁, Δ₂, T, V₁, V₂₁, V₂₂, V₃, V₄, D), magneticstructure))
    update!(FMOAFMMP, Jxy₁=0.5742, JAxy₂=-0.06522, JBxy₂=-0.01386, Jxy₃=-0.2113, Δ₁=-3.745, Δ₂=-2.836)
    update!(FMOAFMMP, V₁=38.0, V₂₁=17.0, V₂₂=7.5, V₃=12.5, V₄=7.0, D=2.2)
    path = ReciprocalPath(lattice.reciprocals, (0, 0, 0)=>(2, 0, 0), length=400)
    afmeb = FMOAFMMP(:EB, EnergyBands(path, collect(1:16)))
    plt = plot(afmeb, xminorticks=10, yminorticks=10, minorgrid=true)
    ylims!(plt, 0.0, 16.0)
    display(plt)
    savefig(plt, "magnon-phonon-hybridization.png")
end