function forward_temporaries(body, Δs)
  # exs = union((common(body, Δ) for Δ in values(Δs))...)
  # filter!(ex -> !(@capture(value(ex), self._) || isconstant(ex)), exs)
  # [ex=>symbol("temp", i) for (i, ex) in enumerate(exs)]
  return Dict()
end

function process_func(ex, params)
  @capture(shortdef(ex), (args__,) -> body_)
  body = il(graphm(body))
  body = map(x -> x in params ? :(self.$x) : x, body)
  Δ = invert(body, @flow(Δ))
  return args, body, Δ
end

function build_type(T, params, temps)
  quote
    type $T
      $(params...)
      $([symbol("Δ", s) for s in params]...)
      $(temps...)
    end
    $T($(params...)) = $T($(params...),
                          $((:(zeros($p)) for p in params)...),
                          $((:nothing for t in temps)...))
  end
end

function build_forward(body, temps)
  forward = IVertex{Any}(Flow.Do())
  for (ex, k) in temps
    k = Expr(:quote, k)
    thread!(forward, @v(setfield!(:self, k, ex)))
  end
  thread!(forward, body)
  cse(forward)
end

function build_backward(Δs, x, params, temps)
  back = IVertex{Any}(Flow.Do())
  tempify(v) = prewalk(v -> haskey(temps, v) ? @v(:(self.$(temps[v]))) : v, v)
  for param in params
    haskey(Δs, :(self.$param)) || continue
    k = symbol("Δ", param)
    ksym = Expr(:quote, k)
    ex = tempify(Δs[:(self.$param)])
    thread!(back, @v(setfield!(:self, ksym, :(self.$k) + ex)))
  end
  thread!(back, tempify(Δs[x]))
  cse(back)
end

function build_update(T, params)
  updates = []
  for p in params
    Δp = symbol("Δ", p)
    push!(updates, :(self.$p += self.$Δp; fill!(self.$Δp, 0)))
  end
  :(update!(self::$T) = $(updates...))
end

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = true || [],
             funcs  = false || []] = groupby(x->isa(x, Symbol), fs)
  @assert length(funcs) == 1
  args, body, Δs = process_func(funcs[1], params)
  @assert length(args) == 1
  temps = forward_temporaries(body, Δs)
  quote
    $(build_type(T, params, collect(values(temps))))
    (self::$T)($(args...),) = $(syntax(build_forward(body, temps)))
    back!(self::$T, Δ, $(args...)) = $(syntax(build_backward(Δs, args[1], params, temps)))
    $(build_update(T, params))
  end |> longdef |> MacroTools.flatten
end

process_type(:(type Sigmoid
  W
  b
  x -> σ(W*x+b)
end)) |> prettify
