module MagnonPhononHybridization

using LinearAlgebra: norm, dot
using StaticArrays: SVector
using QuantumLattices: ID, Operator, AbstractPID, PID, Point, AbstractBond, Bond, NID, SID, FID, Index, OID, Hilbert, Phonon, Fock, Spin, CompositeInternal, SimpleIID
using QuantumLattices: Couplings, Coupling, TermCouplings, TermAmplitude, TermModulate, Term, VectorSpace, VectorSpaceStyle, VectorSpaceCartesian, Table, Metric, CompositeOID
using QuantumLattices: ⊕, ⊗, dimension, dtype, pidtype, rcoord, icoord, couplinginternals, totalspin, wildcard, atol, rtol
using TightBindingApproximation: TBAKind, TBAMatrixRepresentation
using SpinWaveTheory: MagneticStructure, LSWT

import LinearAlgebra: ishermitian
import QuantumLattices: expand, shape, abbr, couplingcenters, optype, add!
import TightBindingApproximation: commutator

export DMHybridization, @dmhybridization_str, MPHMetric

"""
    expand(dmp::Coupling{<:Number, <:Tuple{NID{Symbol}, SID{wildcard, Int, Symbol}}}, bond::Bond, hilbert::Hilbert, info::Val{:DMHybridization}) -> DMPExpand

Expand the default DM magnon-phonon coupling on a given bond.
"""
function expand(dmp::Coupling{<:Number, <:Tuple{NID{Symbol}, SID{wildcard, Int, Symbol}}}, bond::Bond, hilbert::Hilbert, info::Val{:DMHybridization})
    R̂, a = rcoord(bond)/norm(rcoord(bond)), norm(rcoord(bond))
    phonon, spin = couplinginternals(dmp, bond, hilbert, info)
    @assert phonon.ndir==length(R̂) "expand error: mismatched number of directions."
    @assert isapprox(dmp.value, 1, atol=atol, rtol=rtol) "expand error: wrong coefficient of DM magnon-phonon coupling."
    @assert dmp.cid[1].tag=='u' && dmp.cid[2].orbital==1 && spin.norbital==1 "expand error: not supported expansion of DM magnon-phonon coupling."
    return DMPExpand{totalspin(spin)}(totalspin(spin)/a, R̂, (bond.epoint, bond.spoint))
end
struct DMPExpand{S, V<:Number, D, P<:AbstractPID} <: VectorSpace{Operator{V, Tuple{OID{Index{P, NID{Char}}, SVector{D, V}}, OID{Index{P, SID{S, Int, Char}}, SVector{D, V}}}}}
    value::V
    direction::SVector{D, V}
    points::NTuple{2, Point{D, P, V}}
    DMPExpand{S}(value::Number, direction::SVector{D}, points::NTuple{2, Point}) where {S, D} = new{S, typeof(value), D, pidtype(eltype(points))}(value, direction, points)
end
@inline VectorSpaceStyle(::Type{<:DMPExpand}) = VectorSpaceCartesian()
@inline shape(dmp::DMPExpand) = (1:2, 1:2, 1:2, 1:2)
function Operator(index::CartesianIndex{4}, dmp::DMPExpand{S}) where S
    coeff = (-dmp.direction[index[1]]*dmp.direction[index[2]]+(index[1]==index[2] ? 1 : 0))*(index[3]==1 ? 1 : -1)
    oid₁ = OID(Index(dmp.points[index[3]].pid, NID('u', index[1]==1 ? 'x' : 'y')), dmp.points[index[3]].rcoord, dmp.points[index[3]].icoord)
    oid₂ = OID(Index(dmp.points[index[4]].pid, SID{S}(1, index[2]==1 ? 'x' : 'y')), dmp.points[index[4]].rcoord, dmp.points[index[4]].icoord)
    return Operator(dmp.value*coeff, oid₁, oid₂)
end

"""
    DMHybridization(id::Symbol, value::Any, bondkind::Int; amplitude::Union{Function, Nothing}=nothing, modulate::Union{Function, Bool}=false)

The DM Magnon-Phonon coupling term.

Type alias for `Term{:DMHybridization, id, V, Int, C<:TermCouplings, A<:TermAmplitude, M<:TermModulate}`
"""
const DMHybridization{id, V, C<:TermCouplings, A<:TermAmplitude, M<:TermModulate} = Term{:DMHybridization, id, V, Int, C, A, M}
macro dmhybridization_str(::String) Couplings(Coupling(1, ID(NID('u', wildcard), SID{wildcard}(1, wildcard)))) end
@inline function DMHybridization(id::Symbol, value::Any, bondkind::Int;
        amplitude::Union{Function, Nothing}=nothing,
        modulate::Union{Function, Bool}=false
        )
    Term{:DMHybridization}(id, value, bondkind, couplings=dmhybridization"", amplitude=amplitude, modulate=modulate)
end
@inline abbr(::Type{<:DMHybridization}) = :dmp
@inline ishermitian(::Type{<:DMHybridization}) = true
@inline couplingcenters(::Coupling, ::Bond, ::Val{:DMHybridization}) = (1, 2)
@inline function optype(T::Type{<:Term{:DMHybridization}}, H::Type{<:Hilbert}, B::Type{<:AbstractBond})
    V = SVector{dimension(eltype(B)), dtype(eltype(B))}
    I₁ = OID{Index{pidtype(eltype(B)), NID{Char}}, V}
    I₂ = OID{Index{pidtype(eltype(B)), SID{totalspin(filter(SID, valtype(H))), Int, Char}}, V}
    Operator{valtype(T), Tuple{I₁, I₂}}
end

@inline function Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊕, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure)
    return Hilbert(pid=>filter(NID, hilbert[pid])⊕Fock{:b}(norbital=filter(SID, hilbert[pid]).norbital, nspin=1, nnambu=2) for pid in magneticstructure.cell.pids)
end
@inline function Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊗, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure)
    return Hilbert(pid=>filter(NID, hilbert[pid])⊗Fock{:b}(norbital=filter(SID, hilbert[pid]).norbital, nspin=1, nnambu=2) for pid in magneticstructure.cell.pids)
end

struct MPHMetric <: Metric end
@inline Base.valtype(::Type{MPHMetric}, ::Type{<:Index}) = NTuple{4, Int}
function (::MPHMetric)(index::Index)
    if isa(index.iid, FID{:b})
        return (1, index.iid.nambu, index.pid.site, index.iid.orbital)
    elseif isa(index.iid, NID)
        return (2, -Int(index.iid.tag), index.pid.site, Int(index.iid.dir))
    else
        error("not supported index.")
    end
end
@inline Metric(::TBAKind{:BdG}, hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}}}) where K = MPHMetric()
function Table(hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}}}, by::MPHMetric) where K
    result = Index{keytype(hilbert), <:SimpleIID}[]
    for (pid, internal) in hilbert
        for iid in filter(NID, internal)
            push!(result, Index(pid, iid))
        end
        for iid in filter(FID, internal)
            push!(result, Index(pid, iid))
        end
    end
    return Table(result, by)
end
function commutator(k::TBAKind{:BdG}, hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}}}) where K
    m₁ = commutator(k, Hilbert(pid=>filter(FID, internal) for (pid, internal) in hilbert))
    m₂ = commutator(k, Hilbert(pid=>filter(NID, internal) for (pid, internal) in hilbert))
    result = zeros(Complex{Int}, size(m₁)[1]+size(m₂)[1], size(m₁)[1]+size(m₂)[1])
    result[1:size(m₁)[1], 1:size(m₁)[2]] = m₁
    result[(size(m₁)[1]+1):size(result)[1], (size(m₁)[2]+1):size(result)[2]] = -m₂
    return result
end

function add!(dest::Matrix,
        mr::TBAMatrixRepresentation{<:LSWT},
        m::Operator{<:Number, <:Tuple{CompositeOID{<:Index{PID, <:NID}}, CompositeOID{<:Index{PID, <:FID{:b}}}}};
        atol=atol/5,
        kwargs...
        )
    coord = mr.gauge==:rcoord ? rcoord(m) : icoord(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.k, coord)))
    seq₁ = mr.table[m[1].index]
    seq₂ = mr.table[m[2].index]
    dest[seq₁, seq₂] += m.value*phase
    dest[seq₂, seq₁] += m.value'*phase'
    return dest
end

end # module
