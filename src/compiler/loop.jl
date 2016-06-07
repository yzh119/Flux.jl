function delays(v::IVertex)
  ds = []
  Flow.prefor(v) do w
    value(w) == :Delay &&
      push!(ds, w)
  end
  return ds
end

function cut_forward(v::IVertex)
  pushes = map(x->vertex(:push!, vertex(:(self.delay)), v[1]), delays(v))
  isempty(pushes) && return v
  @assert length(pushes) == 1
  v = vertex(Flow.Do(), pushes..., v)
  prewalk(v) do v
    value(v) == :Delay || return v
    il(@flow(pop!(self.delay)))
  end
end
