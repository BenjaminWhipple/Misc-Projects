using NeuralPDE, Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using DifferentialEquations, Plots
import ModelingToolkit: Interval

u0 = [3., 3.]
x0 = 3.
y0 = 3.

t0=0.
tf=30.

# Dynamical systems model parameters 
#alpha, beta, gamma, delta, epsilon, K1, K2
alpha = 1.
beta = .5
gamma = .5
delta = 1.
epsilon = 1.
K1 = 5.
K2 = 5.

# Loss function parameters
c = 1.
a = 1.
b = 1.

# Depreciation rate.
r = 0.06

@parameters x y t
@variables V(..)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)

Dx2 = Differential(x)^2

eq = -Dt(V(t,x,y)) ~ -(1/(4*c))*exp(-r*t)*Dx2(V(t,x,y)) + a*x - b*y
    + Dx(V(t,x,y))*(alpha*x*(1-x/K1)-beta*x*y)
    + Dy(V(t,x,y))*(gamma*x*y - delta*y^2/K2 - epsilon*y)


bcs = [V(tf,x,y)~0.0,
       Dt(V(t,10.0,y))~0,
       Dt(V(t,0.0,y))~0,
       Dt(V(t,x,10.0))~0,
       Dt(V(t,x,0.0))~0]

domains = [t ∈ Interval(t0, tf),
    x ∈ Interval(0.0, 12.0),
    y ∈ Interval(0.0, 12.0)]

# Neural network Specification
dim = 3 # number of dimensions
chain = Lux.Chain(Dense(dim, 50, tanh), Dense(50, 50, tanh), Dense(50, 1))

# Discretization
dx = 0.1

# QuadratureTraining seems to work best.
discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining())

@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [V(t,x,y)])
prob = discretize(pde_system, discretization)

#Callback function
callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# maxiters 50000 works well
@time res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 50000)
print(res)

phi = discretization.phi

ts, xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]

function net_pred(u)
    # u[1] = t, u[2] = x, u[3] = y
    return phi(u,res.u)
end

function net_pred_grad(u)
    # u[1] = t, u[2] = x, u[3] = y
    return gradient(v->phi(v,res.u)[1],u)[1]
end

println(net_pred([1.0,1.0,1.0]))
println(net_pred_grad([1.0,1.0,1.0]))

function control(u)
    # We can recover the control from the HJB equation. u = 1/2c * e^rt * del_x V
    return (1/(2*c))*exp(r*u[1])*net_pred_grad(u)[2]
end

## We notice that bounds on the control can be computed as bounds on the Value derivative (in this case). In principle, this should not be significantly harder to implement, however, it is likely to make the optimization more difficult (as well as destroying the simplifying assumptions we made to prove optimality).

println(control([1.0,1.0,1.0]))

function Test_Model!(du, u, p, t)
    alpha, beta, gamma, delta, epsilon, K1, K2 = p
    x,y = u
    U_control = control([t,x,y])
    #du
    du[1]=dx = alpha*x*(1-x/K1)-beta*x*y - U_control
    du[2]=dy = gamma*x*y - delta*y^2/K2 - epsilon*y
end

# Define callbacks
function condition(u,t,integrator)
    u[1] < 1e-8
end

function affect!(integrator)
    integrator.u[1]=0
end

cb = DiscreteCallback(condition, affect!)

tspan = (t0, tf)
tsteps = t0:0.1:tf

# Make system input vector
#U0 = [control([[0] u0]) u0]

#alpha, beta, gamma, delta, epsilon, K1, K2

p = [alpha, beta, gamma, delta, epsilon, K1, K2]
prob = ODEProblem(Test_Model!, u0, tspan, p)

# This IVP takes quite some time to complete.
@time sol = solve(prob, Rodas5P(),callback=cb,saveat=tsteps,abstol=1e-10,reltol=1e-10)
times = sol.t

p1 = plot(sol.t,sol[1,:],label="x")
p1 = plot!(sol.t,sol[2,:],label="y")

# This loop section is expensive and should be improved.
N = length(sol.t)
controls = []
for i in 1:N
    push!(controls, control([sol.t[i];sol.u[i]]))
end

p2 = plot(sol.t,controls,label="u",color=:red)

plot(p1,p2, layout=(2,1))
savefig("MainPlots.png")


