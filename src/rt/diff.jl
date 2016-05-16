import Flow: isconstant, il, dl, cse, prewalk, graphm, syntax

vertex(a...) = IVertex{Any}(a...)

∇(f, a::IVertex) =
  (v(∇₁(f), a),)

∇(::typeof(+), a::IVertex, b::IVertex) =
  v(1), v(1)

∇(::typeof(-), a::IVertex, b::IVertex) =
  v(1), v(-1)

∇(::typeof(*), a::IVertex, b::IVertex) =
  v(transpose, b), v(transpose, a)

function ∇v(v::IVertex, chain::Vector{IVertex}, out = d())
  if isconstant(v)
    @assert !haskey(out, value(v))
    out[value(v)] = length(chain) == 1 ?
      first(chain) :
      foldl((x, y) -> vertex(*, x, y), chain)
  else
    ∇s = ∇(value(v), inputs(v)...)
    for (v′, ∇′) in zip(inputs(v), ∇s)
      ∇v(v′, (value(∇′) ≠ 1 ? push!(copy(chain), ∇′) : chain), out)
    end
  end
  return out
end

∇v(v::Vertex, chain::Vector) = ∇v(convert(IVertex, v), convert(Vector{IVertex}, chain))

∇v(v::Vertex, ∂::Vertex) = ∇v(v, [∂])

