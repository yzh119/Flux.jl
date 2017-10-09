module Graph

using DataFlow, MacroTools
using DataFlow: graphm, il, constructor, mapconst

export @net

graph(x) = nothing

function graphdef(m, T, args, body)
  v = il(graphm(body, args = args)) |> DataFlow.striplines
  :(Graph.graph($(esc(m))::$(esc(T))) = $(constructor(mapconst(esc, v))))
end

macro net(ex)
  @capture(shortdef(ex), (m_::T_)(args__) = body_) ||
    error("@net requires a forward pass")
  quote
    $(esc(ex))
    $(graphdef(m, T, args, body))
  end
end

end
