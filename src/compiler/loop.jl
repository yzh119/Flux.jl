function cut_forward(v::IVertex)
  pushes = []
  Flow.prefor(v) do w
    value(w) == :Delay &&
      push!(pushes, vertex(:push!, vertex(:(self.delay)), w[1]))
  end
  isempty(pushes) && return v
  @assert length(pushes) == 1
  v = vertex(Flow.Do(), pushes..., v)
  prewalk(v) do v
    value(v) == :Delay || return v
    il(@flow(pop!(self.delay)))
  end
end
