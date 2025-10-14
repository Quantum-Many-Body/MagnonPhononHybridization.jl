module MagnonPhononHybridization

using LinearAlgebra: norm
using QuantumLattices: atol, lazy, plain, rtol
using QuantumLattices: Bond, CoordinatedIndex, CompositeInternal, Coupling, Fock, FockIndex, Index, InternalIndex, InternalProd, InternalSum, Lattice, Neighbors, OneOrMore, Operator, OperatorGenerator, OperatorSum, Pattern, Phonon, PhononIndex, Point, Spin, SpinIndex, Term, TermAmplitude, TermCoupling, VectorSpace, VectorSpaceDirectProducted
using QuantumLattices: ⊕, ⊗, 𝕊, 𝕦, bonds, dimension, icoordinate, nneighbor, rcoordinate, scalartype, totalspin, @pattern
using SpinWaveTheory: HolsteinPrimakoff, MagneticStructure, Magnonic
using StaticArrays: SVector
using TightBindingApproximation: Phononic, Quadratic, Quadraticization, TBAKind

import QuantumLattices: Hilbert, Metric, Table, VectorSpaceStyle, add!, expand, operatortype, shape
import SpinWaveTheory: LSWT
import TightBindingApproximation: commutator

export DMHybridization, LSWT, MagnonPhonon, MagnonPhononCoupled, MPHMetric, SpinPhonon

"""
    SpinPhonon = Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}

Internal spin-phonon space.
"""
const SpinPhonon = Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}

"""
    const MagnonPhonon = Union{Tuple{Phonon, Fock{:b}}, Tuple{Fock{:b}, Phonon}}

Internal magnon-phonon space.
"""
const MagnonPhonon = Union{Tuple{Phonon, Fock{:b}}, Tuple{Fock{:b}, Phonon}}

"""
    expand(dmp::Coupling{<:Number, <:Pattern{<:Tuple{Index{<:PhononIndex{:u}}, Index{<:SpinIndex}}}}, ::Val{:DMHybridization}, bond::Bond, hilbert::Hilbert) -> DMPExpand

Expand the default DM magnon-phonon coupling on a given bond.
"""
function expand(dmp::Coupling{<:Number, <:Pattern{<:Tuple{Index{<:PhononIndex{:u}}, Index{<:SpinIndex}}}}, ::Val{:DMHybridization}, bond::Bond, hilbert::Hilbert)
    @assert isa(dmp.pattern.indexes.sites, NTuple{2, Colon}) "expand error: the `:sites` attributes of the DMHybridization coupling pattern must be a 2-tuple of colons."
    R̂, a = rcoordinate(bond)/norm(rcoordinate(bond)), norm(rcoordinate(bond))
    phonon = filter(dmp.pattern[1].internal, hilbert[bond[1].site])
    spin = filter(dmp.pattern[2].internal, hilbert[bond[2].site])
    @assert phonon.ndirection==length(R̂) "expand error: mismatched number of directions."
    @assert isapprox(dmp.value, 1, atol=atol, rtol=rtol) "expand error: wrong coefficient of DM magnon-phonon coupling."
    return DMPExpand{totalspin(spin)}(totalspin(spin)/a, R̂, (bond.points[2], bond.points[1]))
end
struct DMPExpand{S, V<:Number, D} <: VectorSpace{Operator{V, Tuple{CoordinatedIndex{Index{PhononIndex{:u, Char}, Int}, SVector{D, V}}, CoordinatedIndex{Index{SpinIndex{S, Char}, Int}, SVector{D, V}}}}}
    value::V
    direction::SVector{D, V}
    points::NTuple{2, Point{D, V}}
    DMPExpand{S}(value::Number, direction::SVector{D}, points::NTuple{2, Point}) where {S, D} = new{S, typeof(value), D}(value, direction, points)
end
@inline VectorSpaceStyle(::Type{<:DMPExpand}) = VectorSpaceDirectProducted(:forward)
@inline shape(dmp::DMPExpand) = (1:2, 1:2, 1:2, 1:2)
@inline function Base.convert(::Type{<:Operator}, index::CartesianIndex{4}, dmp::DMPExpand{S}) where S
    coeff = (-dmp.direction[index[1]]*dmp.direction[index[2]]+(index[1]==index[2] ? 1 : 0))*(index[3]==1 ? 1 : -1)
    index₁ = CoordinatedIndex(Index(dmp.points[index[3]].site, PhononIndex{:u}(index[1]==1 ? 'x' : 'y')), dmp.points[index[3]].rcoordinate, dmp.points[index[3]].icoordinate)
    index₂ = CoordinatedIndex(Index(dmp.points[index[4]].site, SpinIndex{S}(index[2]==1 ? 'x' : 'y')), dmp.points[index[4]].rcoordinate, dmp.points[index[4]].icoordinate)
    return Operator(dmp.value*coeff, index₁, index₂)
end

"""
    DMHybridization(id::Symbol, value, bondkind; amplitude::Union{Function, Nothing}=nothing, ismodulatable::Union{Function, Bool}=true)

The DM Magnon-Phonon coupling term.

Type alias for `Term{:DMHybridization, id, V, B, C<:TermCoupling, A<:TermAmplitude}`
"""
const DMHybridization{id, V, B, C<:TermCoupling, A<:TermAmplitude} = Term{:DMHybridization, id, V, B, C, A}
@inline function DMHybridization(
    id::Symbol, value, bondkind;
    amplitude::Union{Function, Nothing}=nothing,
    ismodulatable::Bool=true
)
    return Term{:DMHybridization}(id, value, bondkind, Coupling(@pattern(𝕦(:, α), 𝕊(:, β))), true; amplitude=amplitude, ismodulatable=ismodulatable)
end
@inline function operatortype(::Type{B}, ::Type{H}, ::Type{T}) where {B<:Bond, H<:Hilbert, T<:Term{:DMHybridization}}
    V = SVector{dimension(eltype(B)), scalartype(eltype(B))}
    I₁ = CoordinatedIndex{Index{PhononIndex{:u, Char}, Int}, V}
    I₂ = CoordinatedIndex{Index{SpinIndex{totalspin(filter(SpinIndex, valtype(H))), Char}, Int}, V}
    Operator{valtype(T), Tuple{I₁, I₂}}
end

"""
    MagnonPhononCoupled <: TBAKind{:BdG}

Magnon-phonon coupled quantum lattice system.
"""
struct MagnonPhononCoupled <: TBAKind{:BdG} end

"""
    Hilbert(hilbert::Hilbert{<:InternalSum{<:SpinPhonon}}, magneticstructure::MagneticStructure) -> Hilbert
    Hilbert(hilbert::Hilbert{<:InternalProd{<:SpinPhonon}}, magneticstructure::MagneticStructure) -> Hilbert

Get the hilbert space after the Holstein-Primakoff transformation of a magnon-phonon coupled system.
"""
@inline function Hilbert(hilbert::Hilbert{<:InternalSum{<:SpinPhonon}}, magneticstructure::MagneticStructure)
    return Hilbert(site=>filter(PhononIndex, hilbert[site])⊕Fock{:b}(1, 1) for site=1:length(magneticstructure.cell))
end
@inline function Hilbert(hilbert::Hilbert{<:InternalProd{<:SpinPhonon}}, magneticstructure::MagneticStructure)
    return Hilbert(site=>filter(PhononIndex, hilbert[site])⊗Fock{:b}(1, 1) for site=1:length(magneticstructure.cell))
end

"""
    MPHMetric <: Metric

The metric of the operator indices of a magnon-phonon coupled system.
"""
struct MPHMetric <: Metric end
@inline Base.valtype(::Type{MPHMetric}, ::Type{<:Index}) = NTuple{4, Int}
function (::MPHMetric)(index::Index{<:Union{PhononIndex, FockIndex{:b}}})
    if isa(index.internal, FockIndex{:b})
        return (1, index.internal.nambu, 1, index.site)
    elseif isa(index.internal, PhononIndex{:u})
        return (2, -Int('u'), index.site, Int(index.internal.direction))
    else
        return (2, -Int('p'), index.site, Int(index.internal.direction))
    end
end

"""
    Metric(::MagnonPhononCoupled, ::Hilbert{<:CompositeInternal{<:MagnonPhonon}}) -> MPHMetric

Get the metric of a magnon-phonon coupled system.
"""
@inline Metric(::MagnonPhononCoupled, ::Hilbert{<:CompositeInternal{<:MagnonPhonon}}) = MPHMetric()

"""
    Table(hilbert::Hilbert{<:CompositeInternal{<:MagnonPhonon}}, by::MPHMetric) -> Table

Get the index-sequence table of a magnon-phonon couple system after the Holstein-Primakoff transformation.
"""
function Table(hilbert::Hilbert{<:CompositeInternal{<:MagnonPhonon}}, by::MPHMetric)
    result = Index{<:InternalIndex, Int}[]
    for (site, internal) in hilbert
        for internalindex in filter(PhononIndex{:u}, internal)
            push!(result, Index(site, internalindex))
        end
        for internalindex in filter(PhononIndex{:p}, internal)
            push!(result, Index(site, internalindex))
        end
        for internalindex in filter(FockIndex{:b}, internal)
            push!(result, Index(site, internalindex))
        end
    end
    return Table(result, by)
end

"""
    commutator(::MagnonPhononCoupled, hilbert::Hilbert{<:CompositeInternal{<:MagnonPhonon}}) -> Matrix

Get the commutation relation of the Holstein-Primakoff bosons and phonons.
"""
function commutator(::MagnonPhononCoupled, hilbert::Hilbert{<:CompositeInternal{<:MagnonPhonon}})
    m₁ = commutator(Magnonic(), Hilbert(site=>filter(FockIndex, internal) for (site, internal) in hilbert))
    m₂ = commutator(Phononic(), Hilbert(site=>filter(PhononIndex, internal) for (site, internal) in hilbert))
    result = zeros(Complex{Int}, size(m₁)[1]+size(m₂)[1], size(m₁)[1]+size(m₂)[1])
    result[1:size(m₁)[1], 1:size(m₁)[2]] = m₁
    result[(size(m₁)[1]+1):size(result)[1], (size(m₁)[2]+1):size(result)[2]] = -m₂
    return result
end

"""
    add!(dest::OperatorSum, qf::Quadraticization{MagnonPhononCoupled}, m::Operator{<:Number, <:NTuple{2, CoordinatedIndex{<:Index{<:FockIndex{:b}}}}}; kwargs...) -> typeof(dest)
    add!(dest::OperatorSum, qf::Quadraticization{MagnonPhononCoupled}, m::Operator{<:Number, <:Tuple{CoordinatedIndex{<:Index{<:PhononIndex}}, CoordinatedIndex{<:Index{<:FockIndex{:b}}}}}; kwargs...) -> typeof(dest)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::OperatorSum, qf::Quadraticization{MagnonPhononCoupled}, m::Operator{<:Number, <:NTuple{2, CoordinatedIndex{<:Index{<:FockIndex{:b}}}}}; kwargs...)
    return add!(dest, Quadraticization{Magnonic}(qf.table), m; kwargs...)
end
function add!(dest::OperatorSum, qf::Quadraticization{MagnonPhononCoupled}, m::Operator{<:Number, <:Tuple{CoordinatedIndex{<:Index{<:PhononIndex}}, CoordinatedIndex{<:Index{<:FockIndex{:b}}}}}; kwargs...)
    seq₁, seq₂ = qf.table[m[1]], qf.table[m[2]]
    rcoord, icoord = rcoordinate(m), icoordinate(m)
    add!(dest, Quadratic(m.value, (seq₁, seq₂), rcoord, icoord))
    add!(dest, Quadratic(m.value', (seq₂, seq₁), -rcoord, -icoord))
    return dest
end

"""
    LSWT(
        lattice::Lattice,
        hilbert::Hilbert{<:CompositeInternal{<:SpinPhonon}},
        terms::OneOrMore{Term},
        magneticstructure::MagneticStructure;
        neighbors::Union{Int, Neighbors}=nneighbor(terms)
    )

Construct a LSWT for a magnon-phonon coupled system.
"""
@inline function LSWT(
    lattice::Lattice,
    hilbert::Hilbert{<:CompositeInternal{<:SpinPhonon}},
    terms::OneOrMore{Term},
    magneticstructure::MagneticStructure;
    neighbors::Union{Int, Neighbors}=nneighbor(terms)
)
    H = OperatorGenerator(bonds(magneticstructure.cell, neighbors), hilbert, terms, plain, lazy; half=false)
    hp = HolsteinPrimakoff{valtype(H)}(magneticstructure)
    return LSWT{MagnonPhononCoupled}(lattice, H, hp)
end

end # module
