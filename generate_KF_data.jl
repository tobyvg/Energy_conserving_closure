include("source.jl")
include("NS_FVM_solver.jl")
using JLD

Random.seed!(3);
x= collect(LinRange(-pi,pi,257))
y = collect(LinRange(-pi,pi,257))
#z = collect(LinRange(-pi,pi,257))

#using Base.Threads

#print(nthreads())



UPC = 2
fine_mesh = gen_mesh(x,y,UPC = UPC)

setup = gen_setup(fine_mesh)
0

forcing(x) = sin.(4*x[2])

F = fine_mesh.eval_function(forcing)
F = setup.GS.A_c_s(cat(F,zeros(size(F)),dims = fine_mesh.dims + 1))




max_k = 10
energy_norm = 1
number_of_simulations = 1


KF_rhs = gen_rhs(setup,F,damping = 0.1)

# gen divergence free initial condition
V = gen_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (fine_mesh.UPC,number_of_simulations))
MV = setup.O.M(padding(V,(1,1),circular = true))
p = setup.PS(MV)
Gp = setup.O.G(padding(p,(1,1),circular =true))
V0 = Float32.(V-Gp)



#heatmap(setup.O.M(V0)[:,:,1,1])

t_start = 0
t_end = 100
dt = 0.01
save_every = 1000
pre_allocate = true


t_data,sim_data = simulate(V0,fine_mesh,dt,t_start,t_end,KF_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate)
#0

GC.gc()


t_start = 0
t_end = 100
dt = 0.01
save_every = 5
pre_allocate = true

V0 = sim_data[:,:,:,:,end]
sim_data = 0
t_data = 0

GC.gc()

t_data,sim_data = simulate(V0,fine_mesh,dt,t_start,t_end,KF_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate)

d = save("data/KF/training_data.jld","t",t_data,"V",sim_data,"F",F)



t_start = 0
t_end = 100
dt = 0.01
save_every = 20
pre_allocate = true

V0 = sim_data[:,:,:,:,end]
sim_data = 0
t_data = 0

GC.gc()

t_data,sim_data = simulate(V0,fine_mesh,dt,t_start,t_end,KF_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate)

d = save("data/KF/test_data.jld","t",t_data,"V",sim_data,"F",F)
