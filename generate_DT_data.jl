include("source.jl")
include("NS_FVM_solver.jl")
using JLD

Random.seed!(5);
x= collect(LinRange(-pi,pi,257))
y = collect(LinRange(-pi,pi,257))
#z = collect(LinRange(-pi,pi,257))

#using Base.Threads

#print(nthreads())



UPC = 2
fine_mesh = gen_mesh(x,y,UPC = UPC)

setup = gen_setup(fine_mesh)
0

F= zeros(size(fine_mesh.omega))



max_k = 10
energy_norm = 1
number_of_simulations = 10


DT_rhs = gen_rhs(setup,F,damping = 0)

# gen divergence free initial condition
V = gen_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (fine_mesh.UPC,number_of_simulations))
MV = setup.O.M(padding(V,(1,1),circular = true))
p = setup.PS(MV)
Gp = setup.O.G(padding(p,(1,1),circular =true))
V0 = Float32.(V-Gp)


t_start = 0
t_end = 10
dt = 0.01
save_every = 5
pre_allocate = true




t_data,sim_data = simulate(V0,fine_mesh,dt,t_start,t_end,DT_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate)

d = save("data/DT/training_data.jld","t",t_data,"V",sim_data,"F",F)


# generate divergence free initial condition
V = gen_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (fine_mesh.UPC,number_of_simulations))
MV = setup.O.M(padding(V,(1,1),circular = true))
p = setup.PS(MV)
Gp = setup.O.G(padding(p,(1,1),circular =true))
V0 = Float32.(V-Gp)

t_start = 0
t_end = 10
dt = 0.01
save_every = 20
pre_allocate = true

sim_data = 0
t_data = 0

GC.gc()

t_data,sim_data = simulate(V0,fine_mesh,dt,t_start,t_end,DT_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate)

d = save("data/DT/test_data.jld","t",t_data,"V",sim_data,"F",F)
