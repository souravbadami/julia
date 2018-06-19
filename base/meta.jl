# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
Convenience functions for metaprogramming.
"""
module Meta

using ..CoreLogging

export quot,
       isexpr,
       show_sexpr,
       @dump

quot(ex) = Expr(:quote, ex)

isexpr(@nospecialize(ex), head::Symbol) = isa(ex, Expr) && ex.head === head
isexpr(@nospecialize(ex), heads::Union{Set,Vector,Tuple}) = isa(ex, Expr) && in(ex.head, heads)
isexpr(@nospecialize(ex), heads, n::Int) = isexpr(ex, heads) && length(ex.args) == n


# ---- show_sexpr: print an AST as an S-expression ----

show_sexpr(ex) = show_sexpr(stdout, ex)
show_sexpr(io::IO, ex) = show_sexpr(io, ex, 0)
show_sexpr(io::IO, ex, indent::Int) = show(io, ex)

const sexpr_indent_width = 2

function show_sexpr(io::IO, ex::QuoteNode, indent::Int)
    inner = indent + sexpr_indent_width
    print(io, "(:quote, #QuoteNode\n", " "^inner)
    show_sexpr(io, ex.value, inner)
    print(io, '\n', " "^indent, ')')
end
function show_sexpr(io::IO, ex::Expr, indent::Int)
    inner = indent + sexpr_indent_width
    print(io, '(')
    show_sexpr(io, ex.head, inner)
    for arg in ex.args
        print(io, ex.head === :block ? ",\n"*" "^inner : ", ")
        show_sexpr(io, arg, inner)
    end
    if isempty(ex.args)
        print(io, ",)")
    else
        print(io, (ex.head === :block ? "\n"*" "^indent : ""), ')')
    end
end

"""
    @dump expr

Show every part of the representation of the given expression. Equivalent to
[`dump(:(expr))`](@ref dump).
"""
macro dump(expr)
    return :(dump($(QuoteNode(expr))))
end

"""
    lower(m, x)

Takes the expression `x` and returns an equivalent expression in lowered form
for executing in module `m`.
See also [`code_lowered`](@ref).
"""
lower(m::Module, @nospecialize(x)) = ccall(:jl_expand, Any, (Any, Any), x, m)

"""
    @lower [m] x

Return lowered form of the expression `x` in module `m`.
By default `m` is the module in which the macro is called.
See also [`lower`](@ref).
"""
macro lower(code)
    return :(lower($__module__, $(QuoteNode(code))))
end
macro lower(mod, code)
    return :(lower($(esc(mod)), $(QuoteNode(code))))
end


## interface to parser ##

"""
    ParseError(msg)

The expression passed to the [`parse`](@ref) function could not be interpreted as a valid Julia
expression.
"""
struct ParseError <: Exception
    msg::AbstractString
end

"""
    parse(str, start; greedy=true, raise=true, depwarn=true)

Parse the expression string and return an expression (which could later be passed to eval
for execution). `start` is the index of the first character to start parsing. If `greedy` is
`true` (default), `parse` will try to consume as much input as it can; otherwise, it will
stop as soon as it has parsed a valid expression. Incomplete but otherwise syntactically
valid expressions will return `Expr(:incomplete, "(error message)")`. If `raise` is `true`
(default), syntax errors other than incomplete expressions will raise an error. If `raise`
is `false`, `parse` will return an expression that will raise an error upon evaluation. If
`depwarn` is `false`, deprecation warnings will be suppressed.

```jldoctest
julia> Meta.parse("x = 3, y = 5", 7)
(:(y = 5), 13)

julia> Meta.parse("x = 3, y = 5", 5)
(:((3, y) = 5), 13)
```
"""
function parse(str::AbstractString, pos::Int; greedy::Bool=true, raise::Bool=true,
               depwarn::Bool=true)
    # pos is one based byte offset.
    # returns (expr, end_pos). expr is () in case of parse error.
    bstr = String(str)
    # For now, assume all parser warnings are depwarns
    ex, pos = with_logger(depwarn ? current_logger() : NullLogger()) do
        ccall(:jl_parse_string, Any,
              (Ptr{UInt8}, Csize_t, Int32, Int32),
              bstr, sizeof(bstr), pos-1, greedy ? 1 : 0)
    end
    if raise && isa(ex,Expr) && ex.head === :error
        throw(ParseError(ex.args[1]))
    end
    if ex === ()
        raise && throw(ParseError("end of input"))
        ex = Expr(:error, "end of input")
    end
    return ex, pos+1 # C is zero-based, Julia is 1-based
end

"""
    parse(str; raise=true, depwarn=true)

Parse the expression string greedily, returning a single expression. An error is thrown if
there are additional characters after the first expression. If `raise` is `true` (default),
syntax errors will raise an error; otherwise, `parse` will return an expression that will
raise an error upon evaluation.  If `depwarn` is `false`, deprecation warnings will be
suppressed.

```jldoctest
julia> Meta.parse("x = 3")
:(x = 3)

julia> Meta.parse("x = ")
:($(Expr(:incomplete, "incomplete: premature end of input")))

julia> Meta.parse("1.0.2")
ERROR: Base.Meta.ParseError("invalid numeric constant \\\"1.0.\\\"")
Stacktrace:
[...]

julia> Meta.parse("1.0.2"; raise = false)
:($(Expr(:error, "invalid numeric constant \"1.0.\"")))
```
"""
function parse(str::AbstractString; raise::Bool=true, depwarn::Bool=true)
    ex, pos = parse(str, 1, greedy=true, raise=raise, depwarn=depwarn)
    if isa(ex,Expr) && ex.head === :error
        return ex
    end
    if pos <= ncodeunits(str)
        raise && throw(ParseError("extra token after end of expression"))
        return Expr(:error, "extra token after end of expression")
    end
    return ex
end

# Perform the parts of an inlining pass on Julia IR that are possible when the IR itself is
# pre-type-inference, but the caller has access to the runtime type signature for the
# corresponding method invocation. This function is mainly useful when emitting Julia IR from
# a `@generated function`.
#
# This function is similar to `Core.Compiler.ssa_substitute!`, but works on
# pre-type-inference IR instead of the optimizer's IR, and thus does a couple of extra
# transforms.
#
# Specifically, this function TODO:
# replace slots 1:na with argexprs, static params with spvals, and increment
# other slots by offset.
# eltype(arg_replacements) "<:" Union{SlotNumber,SSAValue,GlobalRef,QuoteNode,Expr(:boundscheck)}
function partially_inline!(x::Vector{Any}, arg_replacements::Vector{Any},
                           @nospecialize(type_signature), static_param_values::Vector{Any},
                           slot_offset::Int, statement_offset::Int,
                           boundscheck::Symbol)
   for i = 1:length(x)
       isassigned(x, i) || continue
       x[i] = _partially_inline!(x[i], arg_replacements, type_signature,
                                 static_param_values, slot_offset,
                                 statement_offset, boundscheck)
   end
   return x
end

function _partially_inline!(@nospecialize(x), arg_replacements::Vector{Any},
                            @nospecialize(type_signature), static_param_values::Vector{Any},
                            slot_offset::Int, statement_offset::Int,
                            boundscheck::Symbol)
    if isa(x, Core.SSAValue)
        return Core.SSAValue(x.id + statement_offset)
    end
    if isa(x, Core.GotoNode)
        return Core.GotoNode(x.id + statement_offset)
    end
    if isa(x, Core.SlotNumber)
        id = x.id
        if 1 <= id <= length(arg_replacements)
            return arg_replacements[id]
        end
        return Core.SlotNumber(id + slot_offset)
    end
    if isa(x, Core.NewvarNode)
        return Core.NewvarNode(_partially_inline!(x.slot, arg_replacements, type_signature,
                                                  static_param_values, slot_offset,
                                                  statement_offset, boundscheck))
    end
    if isa(x, Core.PhiNode)
        partially_inline!(x.values, arg_replacements, type_signature, static_param_values,
                          slot_offset, statement_offset, boundscheck)
        x.edges .+= slot_offset
        return x
    end
    if isa(x, Expr)
        head = x.head
        if head === :static_parameter
            return QuoteNode(static_param_values[x.args[1]])
        elseif head === :cfunction
            @assert !isa(type_signature, UnionAll) || !isempty(spvals)
            if !(x.args[2] isa QuoteNode) # very common no-op
                x.args[2] = _partially_inline!(x.args[2], arg_replacements, type_signature,
                                               static_param_values, slot_offset,
                                               statement_offset, boundscheck)
            end
            x.args[3] = _instantiate_type_in_env(x.args[3], type_signature, static_param_values)
            x.args[4] = svec(Any[_instantiate_type_in_env(argt, type_signature, static_param_values) for argt in x.args[4]]...)
        elseif head === :foreigncall
            @assert !isa(type_signature, UnionAll) || !isempty(static_param_values)
            for i = 1:length(x.args)
                if i == 2
                    x.args[2] = _instantiate_type_in_env(x.args[2], type_signature, static_param_values)
                elseif i == 3
                    x.args[3] = svec(Any[_instantiate_type_in_env(argt, type_signature, static_param_values) for argt in x.args[3]]...)
                elseif i == 4
                    @assert isa((x.args[4]::QuoteNode).value, Symbol)
                elseif i == 5
                    @assert isa(x.args[5], Int)
                else
                    x.args[i] = _partially_inline!(x.args[i], arg_replacements,
                                                   type_signature, static_param_values,
                                                   slot_offset, statement_offset,
                                                   boundscheck)
                end
            end
        elseif head === :boundscheck
            if boundscheck === :propagate
                return x
            elseif boundscheck === :off
                return false
            else
                return true
            end
        elseif head === :gotoifnot
            x.args[1] = _partially_inline!(x.args[1], arg_replacements, type_signature,
                                           static_param_values, slot_offset,
                                           statement_offset, boundscheck)
            x.args[2] += statement_offset
        elseif head === :enter
            x.args[1] += statement_offset
        elseif !is_meta_expr_head(head)
            partially_inline!(x.args, arg_replacements, type_signature, static_param_values,
                              slot_offset, statement_offset, boundscheck)
        end
    end
    return x
end

_instantiate_type_in_env(x, spsig, spvals) = ccall(:jl_instantiate_type_in_env, Any, (Any, Any, Ptr{Any}), x, spsig, spvals)

is_meta_expr_head(head::Symbol) = (head === :inbounds || head === :boundscheck || head === :meta || head === :simdloop)

end # module
