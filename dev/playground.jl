using Pkg;
Pkg.activate("dev/");
using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DecisionTree
using Plots
using TaijaData
using TaijaPlotting

# Counteractual data and model:
n = 3000
data = CounterfactualData(load_moons(n; noise=0.25)...)
X = data.X
M = fit_model(data, :MLP)
fx = predict_label(M, data)
target = 1
factual = 0
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Generic search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)

# T-CREx ###################################################################
ρ = 0.5
τ = 0.9
generator = Generators.TCRExGenerator(; ρ=ρ, τ=τ)

DTExt = Base.get_extension(CounterfactualExplanations, :DecisionTreeExt)

# (a) ##############################

# Surrogate:
model, fitresult = DTExt.grow_surrogate(generator, ce.data, ce.M)
M_sur = CounterfactualExplanations.DecisionTreeModel(model; fitresult=fitresult)
plot(M_sur, data; ms=3, markerstrokewidth=0, size=(500, 500), colorbar=false)

# Extract rules:
R = DTExt.extract_rules(fitresult[1])
println("Expected: ", length(fitresult[1]) * 2 - 1)
println("Observed: ", length(R))
print_tree(fitresult[1])

# Compute feasibility and accuracy:
feas = DTExt.rule_feasibility.(R, (X,))
@assert minimum(feas) >= ρ
acc_factual = DTExt.rule_accuracy.(R, (X,), (fx,), (factual,))
acc_target = DTExt.rule_accuracy.(R, (X,), (fx,), (target,))
@assert all(acc_target .+ acc_factual .== 1.0)

# (b) ##############################
R_max = DTExt.max_valid(R, X, fx, target, τ)
feas_max = DTExt.rule_feasibility.(R_max, (X,))
acc_max = DTExt.rule_accuracy.(R_max, (X,), (fx,), (target,))
plt = plot(data; ms=3, markerstrokewidth=0, size=(500, 500))
p1 = deepcopy(plt)
rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])
for (i, rule) in enumerate(R_max)
    ubx, uby = minimum([rule[1][2], maximum(X[1, :])]),
    minimum([rule[2][2], maximum(X[2, :])])
    lbx, lby = maximum([rule[1][1], minimum(X[1, :])]),
    maximum([rule[2][1], minimum(X[2, :])])
    _feas = round(feas_max[i]; digits=2)
    _n = Int(round(feas_max[i] * n; digits=2))
    _acc = round(acc_max[i]; digits=2)
    @info "Rectangle R$i with feasibility $(_feas) (n≈$(_n)) and accuracy $(_acc)"
    lab = "R$i (ρ̂=$(_feas), τ̂=$(_acc))"
    plot!(
        p1, rectangle(ubx - lbx, uby - lby, lbx, lby); opacity=0.5, color=i + 2, label=lab
    )
end
p1

# (c) ##############################
_grid = DTExt.induced_grid(R_max)
p2 = deepcopy(p1)
function plot_grid!(p)
    for (i, (bounds_x, bounds_y)) in enumerate(_grid)
        lbx, ubx = bounds_x
        lby, uby = bounds_y
        lbx = maximum([lbx, minimum(X[1, :])])
        lby = maximum([lby, minimum(X[2, :])])
        ubx = minimum([ubx, maximum(X[1, :])])
        uby = minimum([uby, maximum(X[2, :])])
        plot!(
            p,
            rectangle(ubx - lbx, uby - lby, lbx, lby);
            fillcolor="black",
            fillalpha=0.1,
            label=nothing,
            lw=2,
        )
    end
end
plot_grid!(p2)
p2

# (d) ##############################
xs = DTExt.prototype.(_grid, (X,); pick_arbitrary=false)
Rᶜ = DTExt.cre.((R_max,), xs, (X,); return_index=true)
p3 = deepcopy(p2)
scatter!(p3, eachrow(hcat(xs...))...; ms=10, label=nothing, color=Rᶜ .+ 2)
p3

# (e) - (f) ########################
bounds = DTExt.partition_bounds(R_max)
tree = DTExt.classify_prototypes(hcat(xs...)', Rᶜ, bounds)
R_final, labels = DTExt.extract_leaf_rules(tree)
p4 = deepcopy(plt)
for (i, rule) in enumerate(R_final)
    ubx, uby = minimum([rule[1][2], maximum(X[1, :])]),
    minimum([rule[2][2], maximum(X[2, :])])
    lbx, lby = maximum([rule[1][1], minimum(X[1, :])]),
    maximum([rule[2][1], minimum(X[2, :])])
    plot!(
        p4,
        rectangle(ubx - lbx, uby - lby, lbx, lby);
        fillalpha=0.5,
        label=nothing,
        color=labels[i] + 2,
    )
end
p4

# (g) ##############################
optimal_rule = apply_tree(tree, vec(x))
p5 = deepcopy(p2)
scatter!(
    p5,
    [x[1]],
    [x[2]];
    ms=10,
    color=2 + optimal_rule,
    label="Local CE (move to R$optimal_rule)",
)
p5
