module Graph

export @net

macro net(ex)
  esc(ex)
end

end
