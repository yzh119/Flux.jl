module Flux

using Lazy, Flow

# Zero Flux Given

export Model, back!, update!

abstract Model
abstract Capacitor <: Model
abstract Activation <: Model

back!(m::Model, ∇) = error("Backprop not implemented for $(typeof(m))")
update!(m::Model, η) = m

include("rt/diff.jl")
include("rt/code.jl")

include("cost.jl")
include("activation.jl")
include("layers/input.jl")
include("layers/dense.jl")
include("layers/sequence.jl")
include("utils.jl")

end # module
