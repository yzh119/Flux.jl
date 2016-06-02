import Flow: isconstant, il, dl, cse, prewalk, graphm, syntax, @v

vertex(a...) = IVertex{Any}(a...)

∇graph(f, ∇, a) = (@v(∇ .* ∇₁(f)(a)),)

∇graph(::typeof(+), ∇, a, b) = ∇, ∇

∇graph(::typeof(-), ∇, a, b) = ∇, @v(-∇)

∇graph(::typeof(*), ∇, a, b) = map(x->@v(∇ * transpose(x)), (b, a))

function ∇graph(v::IVertex, ∇, out = d())
  if isconstant(v)
    @assert !haskey(out, value(v))
    out[value(v)] = il(∇)
  else
    ∇′s = ∇graph(value(v), ∇, inputs(v)...)
    for (v′, ∇′) in zip(inputs(v), ∇′s)
      ∇graph(v′, ∇′, out)
    end
  end
  return out
end

macro derive(ex)
  ∇s = ∇graph(il(graphm(resolve_calls(ex))), @flow(∇))
  v = vertex(Flow.Do(), (@v(Flow.Assign(Symbol("∇", k))(v)) for (k, v) in ∇s)...)
  Expr(:quote, @> v cse syntax prettify)
end
