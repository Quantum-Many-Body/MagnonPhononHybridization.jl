var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MagnonPhononHybridization","category":"page"},{"location":"#MagnonPhononHybridization","page":"Home","title":"MagnonPhononHybridization","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MagnonPhononHybridization.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MagnonPhononHybridization]","category":"page"},{"location":"#MagnonPhononHybridization.DMHybridization","page":"Home","title":"MagnonPhononHybridization.DMHybridization","text":"DMHybridization(id::Symbol, value, bondkind; amplitude::Union{Function, Nothing}=nothing, modulate::Union{Function, Bool}=true)\n\nThe DM Magnon-Phonon coupling term.\n\nType alias for Term{:DMHybridization, id, V, B, C<:TermCoupling, A<:TermAmplitude, M<:TermModulate}\n\n\n\n\n\n","category":"type"},{"location":"#MagnonPhononHybridization.MPHMetric","page":"Home","title":"MagnonPhononHybridization.MPHMetric","text":"MPHMetric <: Metric\n\nThe metric of the operator indices of a magnon-phonon coupled system.\n\n\n\n\n\n","category":"type"},{"location":"#MagnonPhononHybridization.MagnonPhononCoupled","page":"Home","title":"MagnonPhononHybridization.MagnonPhononCoupled","text":"MagnonPhononCoupled <: TBAKind{:BdG}\n\nMagnon-phonon coupled quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumLattices.DegreesOfFreedom.Hilbert-Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.DegreesOfFreedom.CompositeInternal{:⊕, <:Union{Tuple{QuantumLattices.QuantumSystems.Phonon, QuantumLattices.QuantumSystems.Spin}, Tuple{QuantumLattices.QuantumSystems.Spin, QuantumLattices.QuantumSystems.Phonon}}}}, SpinWaveTheory.MagneticStructure}","page":"Home","title":"QuantumLattices.DegreesOfFreedom.Hilbert","text":"Hilbert(hilbert::Hilbert{<:CompositeInternal{:⊕, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure) -> Hilbert\nHilbert(hilbert::Hilbert{<:CompositeInternal{:⊗, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}}}, magneticstructure::MagneticStructure) -> Hilbert\n\nGet the hilbert space after the Holstein-Primakoff transformation of a magnon-phonon coupled system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.DegreesOfFreedom.Metric-Tuple{MagnonPhononCoupled, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.DegreesOfFreedom.CompositeInternal{K, <:Union{Tuple{QuantumLattices.QuantumSystems.Phonon, QuantumLattices.QuantumSystems.Fock}, Tuple{QuantumLattices.QuantumSystems.Fock, QuantumLattices.QuantumSystems.Phonon}}} where K}}","page":"Home","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::MagnonPhononCoupled, ::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}) -> MPHMetric\n\nGet the metric of a magnon-phonon coupled system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.DegreesOfFreedom.Table-Tuple{QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.DegreesOfFreedom.CompositeInternal{K, <:Union{Tuple{QuantumLattices.QuantumSystems.Phonon, QuantumLattices.QuantumSystems.Fock}, Tuple{QuantumLattices.QuantumSystems.Fock, QuantumLattices.QuantumSystems.Phonon}}} where K}, MPHMetric}","page":"Home","title":"QuantumLattices.DegreesOfFreedom.Table","text":"Table(hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}, by::MPHMetric) -> Table\n\nGet the index-sequence table of a magnon-phonon couple system after the Holstein-Primakoff transformation.\n\n\n\n\n\n","category":"method"},{"location":"#SpinWaveTheory.LSWT-Tuple{QuantumLattices.Spatials.Lattice, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.DegreesOfFreedom.CompositeInternal{K, <:Union{Tuple{QuantumLattices.QuantumSystems.Phonon, QuantumLattices.QuantumSystems.Spin}, Tuple{QuantumLattices.QuantumSystems.Spin, QuantumLattices.QuantumSystems.Phonon}}} where K}, Tuple{Vararg{QuantumLattices.DegreesOfFreedom.Term}}, SpinWaveTheory.MagneticStructure}","page":"Home","title":"SpinWaveTheory.LSWT","text":"LSWT(\n    lattice::Lattice,\n    hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Spin}, Tuple{Spin, Phonon}}} where K},\n    terms::Tuple{Vararg{Term}},\n    magneticstructure::MagneticStructure;\n    neighbors::Union{Nothing, Int, Neighbors}=nothing,\n    boundary::Boundary=plain\n)\n\nConstruct a LSWT for a magnon-phonon coupled system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.add!-Tuple{Matrix, TightBindingApproximation.TBAMatrixRepresentation{<:LSWT}, QuantumLattices.QuantumOperators.Operator{<:Number, <:Tuple{QuantumLattices.DegreesOfFreedom.CompositeIndex{<:QuantumLattices.DegreesOfFreedom.Index{Int64, <:QuantumLattices.QuantumSystems.PID}}, QuantumLattices.DegreesOfFreedom.CompositeIndex{<:QuantumLattices.DegreesOfFreedom.Index{Int64, <:QuantumLattices.QuantumSystems.FID{:b}}}}}}","page":"Home","title":"QuantumLattices.add!","text":"add!(dest::Matrix, mr::TBAMatrixRepresentation{<:LSWT}, m::Operator{<:Number, <:Tuple{CompositeIndex{<:Index{Int, <:PID}}, CompositeIndex{<:Index{Int, <:FID{:b}}}}}; atol=atol/5, kwargs...) -> typeof(dest)\n\nGet the matrix representation of an operator and add it to destination.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.expand-Tuple{Val{:DMHybridization}, QuantumLattices.DegreesOfFreedom.Coupling{<:Number, Tuple{QuantumLattices.DegreesOfFreedom.Index{Colon, QuantumLattices.QuantumSystems.PID{Colon}}, QuantumLattices.DegreesOfFreedom.Index{Colon, QuantumLattices.QuantumSystems.SID{:*, Colon}}}}, QuantumLattices.Spatials.Bond, QuantumLattices.DegreesOfFreedom.Hilbert}","page":"Home","title":"QuantumLattices.expand","text":"expand(::Val{:DMHybridization}, dmp::Coupling{<:Number, Tuple{Index{Colon, PID{Colon}}, Index{Colon, SID{wildcard, Colon}}}}, bond::Bond, hilbert::Hilbert) -> DMPExpand\n\nExpand the default DM magnon-phonon coupling on a given bond.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.commutator-Tuple{MagnonPhononCoupled, QuantumLattices.DegreesOfFreedom.Hilbert{<:QuantumLattices.DegreesOfFreedom.CompositeInternal{K, <:Union{Tuple{QuantumLattices.QuantumSystems.Phonon, QuantumLattices.QuantumSystems.Fock}, Tuple{QuantumLattices.QuantumSystems.Fock, QuantumLattices.QuantumSystems.Phonon}}} where K}}","page":"Home","title":"TightBindingApproximation.commutator","text":"commutator(::MagnonPhononCoupled, hilbert::Hilbert{<:CompositeInternal{K, <:Union{Tuple{Phonon, Fock}, Tuple{Fock, Phonon}}} where K}) -> Matrix\n\nGet the commutation relation of the Holstein-Primakoff bosons and phonons.\n\n\n\n\n\n","category":"method"}]
}
