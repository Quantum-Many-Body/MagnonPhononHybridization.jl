```@meta
CurrentModule = MagnonPhononHybridization
```

# [Topological Magnon Polarons in a Multiferroic Material](@id examples)

## Magnon-Polaron bands

The following codes could compute the dispersions of the magnon-polarons in Fe₂Mo₃O₈ by the model proposed in our paper ([Nat. Commun. 14, 6093 (2023)](https://www.nature.com/articles/s41467-023-41791-9)):

```@example Fe₂Mo₃O₈
using QuantumLattices
using TightBindingApproximation
using SpinWaveTheory
using MagnonPhononHybridization
using Plots

const a = 5.773

evenbond(bond) = all(point->iseven(point.site), bond) ? 1 : 0
oddbond(bond) = all(point->isodd(point.site), bond) ? 1 : 0

lattice = Lattice([0.0, 0.0], [0.0, √3/3*a]; vectors=[[1.0, 0.0]*a, [0.5, √3/2]*a])
hilbert = Hilbert(site=>Phonon(2)⊕Spin{2}() for site=1:length(lattice))
magneticstructure = MagneticStructure(
    lattice,
    Dict(site=>(iseven(site) ? [0, 0, 1] : [0, 0, -1]) for site=1:length(lattice))
)

J₁ = Heisenberg(:J₁, 0.5742, 1)
J₂₁ = Heisenberg(:J₂₁, -0.06522, 2; amplitude=oddbond)
J₂₂ = Heisenberg(:J₂₂, -0.01386, 2; amplitude=evenbond)
J₃ = Heisenberg(:J₃, -0.2113, 3)
Δ₁ = SingleIonAnisotropy(:Δ₁, -3.005, 'z'; amplitude=oddbond)
Δ₂ = SingleIonAnisotropy(:Δ₂, -2.250, 'z'; amplitude=evenbond)
T = Kinetic(:T, 0.5)
V₁ = Hooke(:V₁, 36.5, 1)
V₂₁ = Hooke(:V₂₁, 9.1, 2; amplitude=oddbond)
V₂₂ = Hooke(:V₂₂, 7.6, 2; amplitude=evenbond)
V₃ = Hooke(:V₃, 15.5, 3)
V₄ = Hooke(:V₄, 5.5, 4)
D = DMHybridization(:D, 0.8, 1)

Fe₂Mo₃O₈ = Algorithm(
    :Fe₂Mo₃O₈,
    LSWT(
        lattice,
        hilbert,
        (J₁, J₂₁, J₂₂, J₃, Δ₁, Δ₂, T, V₁, V₂₁, V₂₂, V₃, V₄, D),
        magneticstructure
    )
)
path = ReciprocalPath(reciprocals(lattice), hexagon"Γ-M-Γ-K₁-K₂-Γ", length=400)
afmeb = Fe₂Mo₃O₈(:EB, EnergyBands(path, 3:6; tol=10^-6))
plot(
    afmeb;
    xminorticks=10, yminorticks=10, minorgrid=true, title="Magnon-Polaron Bands in Fe₂Mo₃O₈"
)
```

## Berry curvature and Chern number of the magnon-polaron bands
The Berry curvatures and Chern numbers of the magnon-polaron bands could be computed in the reciprocal unitcell:
```@example Fe₂Mo₃O₈
brillouin = BrillouinZone(reciprocals(lattice), 150)
berry = Fe₂Mo₃O₈(:BerryCurvature, BerryCurvature(brillouin, [6, 5]))
plot(berry, color=:RdBu, plot_title="Berry Curvature for the 1st and 2nd Bands in FBZ")
```
Here, k₁ and k₂ denote the coordinates in the reciprocal space along the two reciprocal vectors. Note in the above picture the results for 1st and 2nd bands are shown although `[6, 5]` are used to assign the bands, which arises from the artificial band doubling of the bosonic Bogoliubov transformation to diagonalize the Hamiltonian. To obtain the results for the 3rd and 4th bands:

```@example Fe₂Mo₃O₈
brillouin = BrillouinZone(reciprocals(lattice), 150)
berry = Fe₂Mo₃O₈(:BerryCurvature, BerryCurvature(brillouin, [4, 3]))
plot(berry, color=:RdBu, plot_title="Berry Curvature for the 3rd and 4th Bands in FBZ")
```

The Berry curvatures can also be computed in a reciprocal zone beyond the reciprocal unitcell, e.g., for the 3rd and 4th bands:
```@example Fe₂Mo₃O₈
b₁, b₂ = 4*pi/√3a*[1.0, 0.0], 4*pi/√3a*[0.0, 1.0]
reciprocalzone = ReciprocalZone([b₁, b₂], -1.0=>1.0, -1.0=>1.0; length=301)
berry = Fe₂Mo₃O₈(:BerryCurvature, BerryCurvature(reciprocalzone, [4, 3]))
plot(berry, color=:RdBu, plot_title="Berry Curvature in extended BZ")
```
