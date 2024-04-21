using MagnonPhononHybridization
using Plots: plot, ylims!, savefig
using QuantumLattices
using SpinWaveTheory
using TightBindingApproximation

@testset "MagnonPhononHybridization.jl" begin
    term = DMHybridization(:dmp, 2.0, 1)
    bond = Bond(1, Point(2, [0.5, 0.5], [0.0, 0.0]), Point(1, [0.0, 0.0], [0.0, 0.0]))
    operators = Operators(
        Operator(√2/2, CompositeIndex(Index(2, PID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(1, PID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(1, PID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(2, PID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(1, PID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(1, PID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(2, PID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(1, PID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(2, PID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(1, PID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, CompositeIndex(Index(2, PID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(1, PID('u', 'x')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('y')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(2, PID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('x')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(2, PID('u', 'y')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('y')), [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(2, PID('u', 'x')), [0.5, 0.5], [0.0, 0.0]), CompositeIndex(Index(1, SID{1//2}('x')), [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, CompositeIndex(Index(1, PID('u', 'y')), [0.0, 0.0], [0.0, 0.0]), CompositeIndex(Index(2, SID{1//2}('x')), [0.5, 0.5], [0.0, 0.0]))
    )
    @test expand(term, bond, Hilbert(site=>Phonon(2)⊕Spin{1//2}() for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Spin{1//2}()⊕Phonon(2) for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Phonon(2)⊗Spin{1//2}() for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Spin{1//2}()⊗Phonon(2) for site in [bond[1].site, bond[2].site])) ≈ operators
end

@testset "plot" begin
    a = 5.773
    c = 10.057

    lattice = Lattice(
        [0.0, 0.0, 0.0], [0.0, √3/3*a, 0.0], [0.0, √3/3*a, c/2], [0.0, 0.0, c/2];
        name=Symbol("Stacked-Honeycomb"),
        vectors=[[1.0, 0.0, 0.0]*a, [0.5, √3/2, 0.0]*a, [0.0, 0.0, 1.0]*c]
        )
    neighbors=Neighbors(0=>0.0, 1=>√(a^2/3), 2=>a, 3=>√(4a^2/3), 4=>c/2)
    evenbond(bond) = all(point->iseven(point.site), bond) ? 1 : 0
    oddbond(bond) = all(point->isodd(point.site), bond) ? 1 : 0
    hilbertₘₚ = Hilbert(site=>Phonon(3)⊕Spin{2}() for site=1:length(lattice))
    magneticstructure = MagneticStructure(lattice, Dict(site=>(site%4∈(1, 0) ? [0, 0, 1] : [0, 0, -1]) for site=1:length(lattice)))

    Jxy₁ = SpinTerm(:Jxy₁, 0.5742, 1, MatrixCoupling(:, SID, Ising"x"+Ising"y"))
    JAxy₂ = SpinTerm(:JAxy₂, -0.06522, 2, MatrixCoupling(:, SID, Ising"x"+Ising"y"); amplitude=oddbond)
    JBxy₂ = SpinTerm(:JBxy₂, -0.01386, 2, MatrixCoupling(:, SID, Ising"x"+Ising"y"); amplitude=evenbond)
    Jxy₃ = SpinTerm(:Jxy₃, -0.2113, 3, MatrixCoupling(:, SID, Ising"x"+Ising"y"))
    Δ₁ = SingleIonAnisotropy(:Δ₁, -3.745, 'z'; amplitude=oddbond)
    Δ₂ = SingleIonAnisotropy(:Δ₂, -2.836, 'z'; amplitude=evenbond)
    h = Zeeman(:h, 0.2, 'z')
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 38.0, 1)
    V₂₁ = Hooke(:V₂₁, 17.0, 2; amplitude=oddbond)
    V₂₂ = Hooke(:V₂₂, 7.5, 2; amplitude=evenbond)
    V₃ = Hooke(:V₃, 12.5, 3)
    V₄ = Hooke(:V₄, 7.0, 4)
    D = DMHybridization(:D, 2.2, 1, amplitude=bond->all(point->point.site%4∈(1, 2), bond) ? 1 : -1)

    FMOAFMMP = Algorithm(:FMOAFMMP, LSWT(lattice, hilbertₘₚ, (Jxy₁, JAxy₂, JBxy₂, Jxy₃, Δ₁, Δ₂, h, T, V₁, V₂₁, V₂₂, V₃, V₄, D), magneticstructure; neighbors=neighbors))
    path = ReciprocalPath(reciprocals(lattice), (0, 0, 0)=>(2, 0, 0), length=400)
    afmeb = FMOAFMMP(:EB, EnergyBands(path, 1:16; atol=10^-8))
    plt = plot(afmeb, xminorticks=10, yminorticks=10, minorgrid=true)
    ylims!(plt, 0.0, 16.0)
    display(plt)
    savefig(plt, "magnon-phonon-hybridization.png")
end