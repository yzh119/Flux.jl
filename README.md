# Flux

Flux is an experimental machine perception / ANN library for Julia.

Under the hood, Flux is heavily based on [Torch's](http://torch.ch) elegant model – "modules" are simple Julia objects that conform to an interface for doing efficient forward and backward passes. The Julia advantage is that it allows you to write very efficient kernels without having to "drop down" to unwieldy Cxx interop.

The more interesting part is that we also heavily integrate with the [Flow.jl](https://github.com/MikeInnes/Flow.jl) dataflow language. Because it's easy for us to optimise these graph-based programs, we can write the functionality for new modules in a way that's close to the mathematical form while still being very flexible and efficient.

In future we'd also like to take advantage of the graph model for things like more runtime optimisations (e.g. making small-matrix operations really fast), parallelism, and output to other runtimes like TensorFlow.

Flux can therefore accelerate the research stage of testing new architectures and ideas,
as well as the process of getting those models into production.
