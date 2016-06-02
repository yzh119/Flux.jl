function forward_temporaries(body, ∇s)
  exs = union((common(body, ∇) for ∇ in values(∇s))...)
  filter!(ex -> !@capture(value(ex), self._), exs)
  [ex=>symbol("temp", i) for (i, ex) in enumerate(exs)]
end

resolve_calls(ex) = ex

function resolve_calls(ex::Expr)
  @capture(ex, f_(a__)) ?
    Expr(:call, eval(current_module(), f), map(resolve_calls, a)...) :
    Expr(ex.head, map(resolve_calls, ex.args))
end

function process_func(ex, params)
  @capture(shortdef(ex), (args__,) -> body_)
  body = il(graphm(resolve_calls(body)))
  body = map(x -> x in params ? :(self.$x) : x, body)
  ∇ = ∇graph(body, @flow(∇))
  return args, body, ∇
end

function build_type(T, params, temps)
  quote
    type $T
      $(params...)
      $([symbol("∇", s) for s in params]...)
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

function build_backward(∇s, x, params, temps)
  back = IVertex{Any}(Flow.Do())
  tempify(v) = prewalk(v -> haskey(temps, v) ? @v(:(self.$(temps[v]))) : v, v)
  for param in params
    k = symbol("∇", param)
    ksym = Expr(:quote, k)
    ex = tempify(∇s[:(self.$param)])
    thread!(back, @v(setfield!(:self, ksym, :(self.$k) + ex)))
  end
  thread!(back, tempify(∇s[x]))
  cse(back)
end

function build_update(T, params)
  updates = []
  for p in params
    ∇p = symbol("∇", p)
    push!(updates, :(self.$p += self.$∇p; fill!(self.$∇p, 0)))
  end
  :(update!(self::$T) = $(updates...))
end

function process_type(ex)
  @capture(ex, type T_ fs__ end)
  @destruct [params = true, funcs = false] = groupby(x->isa(x, Symbol), fs)
  @assert length(funcs) == 1
  args, body, ∇s = process_func(funcs[1], params)
  @assert length(args) == 1
  temps = forward_temporaries(body, ∇s)
  ∇s
  quote
    $(build_type(T, params, collect(values(temps))))
    (self::$T)($(args...),) = $(syntax(build_forward(body, temps)))
    back!(self::$T, ∇) = $(syntax(build_backward(∇s, args[1], params, temps)))
    $(build_update(T, params))
  end |> longdef |> MacroTools.flatten
end

process_type(:(type Sigmoid
  W
  b
  x -> σ(W*x+b)
end)) |> prettify
