module Graph

export @net

graph(x) = nothing

macro net(ex)
  esc(ex)
end

end
