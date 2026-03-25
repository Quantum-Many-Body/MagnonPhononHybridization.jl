using MagnonPhononHybridization
using QuantumLattices
using SpinWaveTheory
using TightBindingApproximation
import CairoMakie as Makie
import Plots

@time @testset "MagnonPhononHybridization.jl" begin
    term = DMHybridization(:dmp, 2.0, 1)
    bond = Bond(1, Point(2, [0.5, 0.5], [0.0, 0.0]), Point(1, [0.0, 0.0], [0.0, 0.0]))
    operators = Operators(
        Operator(√2/2, 𝕦(2, 'x', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(2, 'y', [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, 𝕦(1, 'x', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(2, 'x', [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, 𝕦(1, 'x', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(1, 'x', [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, 𝕦(2, 'x', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(1, 'y', [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, 𝕦(1, 'y', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(1, 'y', [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(1, 'y', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(1, 'x', [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, 𝕦(2, 'y', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(2, 'x', [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(1, 'x', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(2, 'y', [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(2, 'y', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(1, 'y', [0.0, 0.0], [0.0, 0.0])),
        Operator(√2/2, 𝕦(1, 'y', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(2, 'y', [0.5, 0.5], [0.0, 0.0])),
        Operator(√2/2, 𝕦(2, 'y', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(1, 'x', [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(1, 'x', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(1, 'y', [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(2, 'x', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(2, 'x', [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(2, 'y', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(2, 'y', [0.5, 0.5], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(2, 'x', [0.5, 0.5], [0.0, 0.0]), 𝕊{1//2}(1, 'x', [0.0, 0.0], [0.0, 0.0])),
        Operator(-√2/2, 𝕦(1, 'y', [0.0, 0.0], [0.0, 0.0]), 𝕊{1//2}(2, 'x', [0.5, 0.5], [0.0, 0.0]))
    )
    @test expand(term, bond, Hilbert(site=>Phonon(2)⊕Spin{1//2}() for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Spin{1//2}()⊕Phonon(2) for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Phonon(2)⊗Spin{1//2}() for site in [bond[1].site, bond[2].site])) ≈ operators
    @test expand(term, bond, Hilbert(site=>Spin{1//2}()⊗Phonon(2) for site in [bond[1].site, bond[2].site])) ≈ operators
end

@time @testset "utilities" begin
    lattice = Lattice([0.0, 0.0], [1.0, 0.0]; vectors=[[1.0, 1.0], [1.0, -1.0]])
    magneticstructure = MagneticStructure(lattice, Dict(site=>iseven(site) ? [0, 0, 1] : [0, 0, -1] for site=1:length(lattice)))
    @test Hilbert(Hilbert(Phonon(2)⊕Spin{2}(), length(lattice)), magneticstructure) == Hilbert(Phonon(2)⊕Fock{:b}(1, 1), length(lattice))
    @test Hilbert(Hilbert(Phonon(2)⊗Spin{2}(), length(lattice)), magneticstructure) == Hilbert(Phonon(2)⊗Fock{:b}(1, 1), length(lattice))

    hilbert = Hilbert(Hilbert(Phonon(2)⊗Spin{2}(), length(lattice)), magneticstructure)
    metric = Metric(MagnonPhononCoupled(), hilbert)
    @test valtype(typeof(metric), Index) == NTuple{4, Int}
    @test Table(hilbert, metric) == Table(
        [   𝕦(1, 'x'), 𝕦(1, 'y'), 𝕦(2, 'x'), 𝕦(2, 'y'),
            𝕡(1, 'x'), 𝕡(1, 'y'), 𝕡(2, 'x'), 𝕡(2, 'y'),
            𝕒(1, 1, 1), 𝕒⁺(1, 1, 1), 𝕒(2, 1, 1), 𝕒⁺(2, 1, 1)
        ],
        metric
    )

    @test commutator(MagnonPhononCoupled(), hilbert) == [
        1 0 0 0     0 0 0 0     0 0 0 0;
        0 1 0 0     0 0 0 0     0 0 0 0;
        0 0 -1 0    0 0 0 0     0 0 0 0;
        0 0 0 -1    0 0 0 0     0 0 0 0;

        0 0 0 0     0 0 0 0     1im 0 0 0;
        0 0 0 0     0 0 0 0     0 1im 0 0;
        0 0 0 0     0 0 0 0     0 0 1im 0;
        0 0 0 0     0 0 0 0     0 0 0 1im;

        0 0 0 0     -1im 0 0 0  0 0 0 0;
        0 0 0 0     0 -1im 0 0  0 0 0 0;
        0 0 0 0     0 0 -1im 0  0 0 0 0;
        0 0 0 0     0 0 0 -1im  0 0 0 0;
    ]
end

@time @testset "plot" begin
    a = 5.773
    c = 10.057

    lattice = Lattice(
        [0.0, 0.0, 0.0], [0.0, √3/3*a, 0.0], [0.0, √3/3*a, c/2], [0.0, 0.0, c/2];
        vectors=[[1.0, 0.0, 0.0]*a, [0.5, √3/2, 0.0]*a, [0.0, 0.0, 1.0]*c]
        )
    neighbors=Neighbors(0=>0.0, 1=>√(a^2/3), 2=>a, 3=>√(4a^2/3), 4=>c/2)
    evenbond(bond) = all(point->iseven(point.site), bond) ? 1 : 0
    oddbond(bond) = all(point->isodd(point.site), bond) ? 1 : 0
    hilbertₘₚ = Hilbert(site=>Phonon(3)⊕Spin{2}() for site=1:length(lattice))
    magneticstructure = MagneticStructure(lattice, Dict(site=>(site%4∈(1, 0) ? [0, 0, 1] : [0, 0, -1]) for site=1:length(lattice)))

    J₁ = Heisenberg(:J₁, 0.5742, 1)
    JA₂ = Heisenberg(:J₂₁, -0.06522, 2; amplitude=oddbond)
    JB₂ = Heisenberg(:J₂₂, -0.01386, 2; amplitude=evenbond)
    J₃ = Heisenberg(:J₃, -0.2113, 3)
    Δ₁ = SingleIonAnisotropy(:Δ₁, -3.005, 'z'; amplitude=oddbond)
    Δ₂ = SingleIonAnisotropy(:Δ₂, -2.250, 'z'; amplitude=evenbond)
    h = Zeeman(:h, 0.2, 'z')
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 38.0, 1)
    V₂₁ = Hooke(:V₂₁, 17.0, 2; amplitude=oddbond)
    V₂₂ = Hooke(:V₂₂, 7.5, 2; amplitude=evenbond)
    V₃ = Hooke(:V₃, 12.5, 3)
    V₄ = Hooke(:V₄, 7.0, 4)
    D = DMHybridization(:D, 2.2, 1, amplitude=bond->all(point->point.site%4∈(1, 2), bond) ? 1 : -1)

    FMOAFMMP = Algorithm(:FMOAFMMP, LSWT(lattice, hilbertₘₚ, (J₁, JA₂, JB₂, J₃, Δ₁, Δ₂, h, T, V₁, V₂₁, V₂₂, V₃, V₄, D), magneticstructure; neighbors=neighbors))
    path = ReciprocalPath(reciprocals(lattice), (0, 0, 0)=>(2, 0, 0), length=400)
    afmeb = FMOAFMMP(:EB, EnergyBands(path, 1:16); tol=10^-6)
    Plots.savefig(Plots.plot(afmeb; ylims=(0.0, 16.0)), "Plots-magnon-phonon-hybridization.png")
    Makie.save("Makie-magnon-phonon-hybridization.png", Makie.plot(afmeb; limits=(nothing, nothing, 0.0, 16.0)))
end
