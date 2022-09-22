module MagnonPhononHybridization

using LinearAlgebra: dot, norm
using QuantumLattices: ⊕, ⊗, dimension, dtype
using QuantumLattices: plain, wildcard, Boundary, CompositeIndex, CompositeInternal, Coupling, Hilbert, Index, Metric, SimpleIID, Table, Term, TermAmplitude, TermCoupling, TermModulate
using QuantumLattices: OperatorGenerator
using QuantumLattices: Operator
using QuantumLattices: FID, Fock, Phonon, PID, SID, Spin, totalspin
using QuantumLattices: Bond, Lattice, Neighbors, Point, bonds, icoordinate, rcoordinate
using QuantumLattices: atol, rtol, VectorSpace, VectorSpaceCartesian, VectorSpaceStyle
using SpinWaveTheory: HPTransformation, MagneticStructure, Magnonic
using StaticArrays: SVector
using TightBindingApproximation: Phononic, TBAKind, TBAMatrixRepresentation

import QuantumLattices: add!, expand, optype, shape
import SpinWaveTheory: LSWT
import TightBindingApproximation: commutator

export DMHybridization, LSWT, MagnonPhononCoupled, MPHMetric

"""
    expand(::Val{:DMHybridization}, dmp::Coupling{<:Number, Tuple{Index{Colon, PID{Colon}}, Index{Colon, SID{wildcard, Colon}}}}, bond::Bond, hilbert::Hilbert) -> DMPExpand

Expand the default DM magnon-phonon coupling on a given bond.
"""
function expand(::Val{:DMHybridization}, dmp::Coupling{<:Number, Tuple{Index{Colon, PID{Colon}}, Index{Colon, SID{wildcard, Colon}}}}, bond::Bond, hilbert::Hilbert)
    R̂, a = rcoordinate(bond)/norm(rcoordinate(bond)), norm(rcoordinate(bond))
    phonon = filter(dmp.indexes[1].iid, hilbert[bond[1].site])
    spin = filter(dmp.indexes[2].iid, hilbert[bond[2].site])
    @assert phonon.ndirection==length(R̂) "expand error: mismatched number of directions."
    @assert isapprox(dmp.value, 1, atol=atol, rtol=rtol) "expand error: wrong coefficient of DM magnon-phonon coupling."
    @assert dmp.indexes[1].iid.tag=='u' "expand error: not supported expansion of DM magnon-phonon coupling."
    return DMPExpand{totalspin(spin)}(totalspin(spin)/a, R̂, (bond.points[2], bond.points[1]))
end
struct DMPExpand{S, V<:Number, D} <: VectorSpace{Operator{V, Tuple{CompositeIndex{Index{Int, PID{Char}}, SVector{D, V}}, CompositeIndex{Index{Int, SID{S, Char}}, SVector{D, V}}}}}
    value::V
    direction::SVector{D, V}
    points::NTuple{2, Point{D, V}}
    DMPExpand{S}(value::Number, direction::SVector{D}, points::NTuple{2, Point}) where {S, D} = new{S, typeof(value), D}(value, direction, points)
end
@inline VectorSpaceStyle(::Type{<:DMPExpand}) = VectorSpaceCartesian()
@inline shape(dmp::DMPExpand) = (1:2, 1:2, 1:2, 1:2)
function Operator(index::CartesianIndex{4}, dmp::DMPExpand{S}) where S
    coeff = (-dmp.direction[index[1]]*dmp.direction[index[2]]+(index[1]==index[2] ? 1 : 0))*(index[3]==1 ? 1 : -1)
    index₁ = CompositeIndex(Index(dmp.points[index[3]].site, PID('u', index[1]==1 ? 'x' : 'y')), dmp.points[index[3]].rcoordinate, dmp.points[index[3]].icoordinate)
    index₂ = CompositeIndex(Index(dmp.points[index[4]].site, SID{S}(index[2]==1 ? 'x' : 'y')), dmp.points[index[4]].rcoordinate, dmp.points[index[4]].icoordinate)
    return Operator(dmp.value*coeff, index₁, index₂)
end

"""
    DMHybridization(id::Symbol, value, bondkind; amplitude::Union{Function, Nothing}=nothing, modulate::Union{Function, Bool}=true)

The DM Magnon-Phonon coupling term.

Type alias for `Term{:DMHybridization, id, V, B, C<:TermCoupling, A<:TermAmplitude, M<:TermModulate}`
"""
const DMHybridization{id, V, B, C<:TermCoupling, A<:TermAmplitude, M<:TermModulate} = Term{:DMHybridization, id, V, B, C, A, M}
@inline function DMHybridization(
    id::Symbol, value, bondkind;
    amplitude::Union{Function, Nothing}=nothing,
    modulate::Union{Function, Bool}=true
)
    return Term{:DMHybridization}(id, value, bondkind, Coupling(Index(:, PID('u', :)), Index(:, SID(:))), true; amplitude=amplitude, modulate=modulate)
end
@inline function optype(T::Type{<:Term{:DMHybridization}}, H::Type{<:Hilbert}, B::Type{<:Bond})
    V = SVector{dimension(eltype(B)), dtype(eltype(B))}
    I₁ = CompositeIndex{Index{Int, PID{Char}}, V}
    I₂ = CompositeIndex{Index{Int, SID{totalspin(filter(SID, valtype(H))), Char}}, V}
    Operator{valtype(T), Tuple{I₁, I₂}}
end

"""
    MagnonPhononCoupled <: TBAKind{:BdG}

Magnon-phonon coupled quantum lattice system.
"""
struct MagnonPhononCoupled <: TBAKind{:BdG} end

"""
    Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊕, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure) -> Hilbert
    Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊗, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure) -> Hilbert

Get the hilbert space after the Holstein-Primakoff transformation of a magnon-phonon coupled system.
"""
@inline function Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊕, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure)
    return Hilbert(site=>filter(PID, hilbert[site])⊕Fock{:b}(1, 1) for site=1:length(magneticstructure.cell))
end
@inline function Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊗, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure)
    return Hilbert(site=>filter(PID, hilbert[site])⊗Fock{:b}(1, 1) for site=1:length(magneticstructure.cell))
end

"""
    MPHMetric <: Metric

The metric of the operator indices of a magnon-phonon coupled system.
"""
struct MPHMetric <: Metric end
@inline Base.valtype(::Type{MPHMetric}, ::Type{<:Index}) = NTuple{4, Int}
function (::MPHMetric)(index::Index)
    if isa(index.iid, FID{:b})
        return (1, index.iid.nambu, 1, index.site)
    elseif isa(index.iid, PID)
        return (2, -Int(index.iid.tag), index.site, Int(index.iid.direction))
    else
        error("not supported index.")
    end
end

"""
    Metric(::MagnonPhononCoupled, ::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}) -> MPHMetric

Get the metric of a magnon-phonon coupled system.
"""
@inline Metric(::MagnonPhononCoupled, ::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}) = MPHMetric()

"""
    Table(hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}, by::MPHMetric) -> Table

Get the index-sequence table of a magnon-phonon couple system after the Holstein-Primakoff transformation.
"""
function Table(hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}, by::MPHMetric)
    result = Index{Int, <:SimpleIID}[]
    for (site, internal) in hilbert
        for iid in filter(PID, internal)
            push!(result, Index(site, iid))
        end
        for iid in filter(FID, internal)
            push!(result, Index(site, iid))
        end
    end
    return Table(result, by)
end

"""
    commutator(::MagnonPhononCoupled, hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}) -> Matrix

Get the commutation relation of the Holstein-Primakoff bosons and phonons.
"""
function commutator(::MagnonPhononCoupled, hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K})
    m₁ = commutator(Magnonic(), Hilbert(site=>filter(FID, internal) for (site, internal) in hilbert))
    m₂ = commutator(Phononic(), Hilbert(site=>filter(PID, internal) for (site, internal) in hilbert))
    result = zeros(Complex{Int}, size(m₁)[1]+size(m₂)[1], size(m₁)[1]+size(m₂)[1])
    result[1:size(m₁)[1], 1:size(m₁)[2]] = m₁
    result[(size(m₁)[1]+1):size(result)[1], (size(m₁)[2]+1):size(result)[2]] = -m₂
    return result
end

"""
    add!(dest::Matrix, mr::TBAMatrixRepresentation{<:LSWT}, m::Operator{<:Number, <:Tuple{CompositeIndex{<:Index{Int, <:PID}}, CompositeIndex{<:Index{Int, <:FID{:b}}}}}; atol=atol/5, kwargs...) -> typeof(dest)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::Matrix, mr::TBAMatrixRepresentation{<:LSWT}, m::Operator{<:Number, <:Tuple{CompositeIndex{<:Index{Int, <:PID}}, CompositeIndex{<:Index{Int, <:FID{:b}}}}}; atol=atol/5, kwargs...)
    coord = mr.gauge==:rcoordinate ? rcoordinate(m) : icoordinate(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coord)))
    seq₁ = mr.table[m[1].index]
    seq₂ = mr.table[m[2].index]
    dest[seq₁, seq₂] += m.value*phase
    dest[seq₂, seq₁] += m.value'*phase'
    return dest
end

"""
    LSWT(
        lattice::Lattice,
        hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}} where K},
        terms::Tuple{Vararg{Term}},
        magneticstructure::MagneticStructure;
        neighbors::Union{Nothing, Int, Neighbors}=nothing,
        boundary::Boundary=plain
    )

Construct a LSWT for a magnon-phonon coupled system.
"""
@inline function LSWT(
    lattice::Lattice,
    hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}} where K},
    terms::Tuple{Vararg{Term}},
    magneticstructure::MagneticStructure;
    neighbors::Union{Nothing, Int, Neighbors}=nothing,
    boundary::Boundary=plain
)
    isnothing(neighbors) && (neighbors=maximum(term->term.bondkind, terms))
    H = OperatorGenerator(terms, bonds(magneticstructure.cell, neighbors), hilbert; half=false, boundary=boundary)
    hp = HPTransformation{valtype(H)}(magneticstructure)
    return LSWT{MagnonPhononCoupled}(lattice, H, hp)
end

end # module
