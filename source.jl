using LinearAlgebra
using Plots
using SparseArrays


using FFTW
#include("source.jl")

using Distributions
using JLD

using Flux
using Random
using LaTeXStrings
using ProgressBars
using Zygote
stop_gradient(f) = f()
Zygote.@nograd stop_gradient


struct mesh_struct
    dims # 1D/2D
    N # grid resoluiont
    x # coordinates
    x_edges # edges
    omega # mass matrix
    eval_function # evaluate function on the grid
    ip # computes inner-product
    integ # integral on the grid
    UPC # unknows per grid cell
end



function construct_k(N)
    dims = length(N)
    k = [fftfreq(i,i) for i in N]

    some_ones = ones(N)
    k_mats = some_ones .* k[1]

    k_mats = reshape(k_mats,(size(k_mats)...,1))

    for i in 2:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = i
        permuted_dims[i] = 1

        k_mat = permutedims(k[i] .* permutedims(some_ones,permuted_dims),permuted_dims)
        k_mats = cat(k_mats,k_mat,dims = dims + 1)
    end
    return k_mats
end


function construct_spectral_filter(k_mats,max_k)
    filter = ones(size(k_mats)[1:end-1])
    N = size(k_mats)[1:end-1]
    dims = length(N)
    loop_over = gen_permutations(N)
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        k = k_mats[i...,:]
        if sqrt(sum(k.^2)) >= max_k
            filter[i...] = 0
        end
    end
    return filter
end

function generate_random_field(N,max_k;norm = 1,samples = (1,1))
    dims = length(N)
    k = construct_k(N)
    filter = construct_spectral_filter(k,max_k)
    coefs = (rand(Uniform(-1,1),(N...,samples...)) + rand(Uniform(-1,1),(N...,samples...)) * (0+1im))

    result = real.(ifft(filter .* coefs,collect(1:dims)))
    sqrt_energies = sqrt.(1/prod(N) .* sum(result.^2,dims = collect(1:dims)))
    result ./= sqrt_energies
    result .*= norm
    return result
end





function gen_mesh(x,y = nothing, z = nothing;UPC=1)
    if y != nothing
        if z != nothing
            x = [z,y,x]
        else
            x = [y,x]
        end
    else
        if length(size(x[1])) <= 0
            x = [x]
        end
    end

    #print(x[1])
    mid_x = [ [(i[j] + i[j+1])/2 for j in 1:(size(i)[1]-1)] for i in x]

    dx = [ [(i[j+1] - i[j]) for j in 1:(size(i)[1]-1)] for i in x]


    sub_grid = ones([size(i)[1] for i in mid_x]...)
    omega = ones([size(i)[1] for i in mid_x]...)


    sub_grids = []

    dims = size(x)[1]


    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1


        omega = permutedims(dx[i] .*  permutedims(omega,permuted_dims),permuted_dims)

        push!(sub_grids,permutedims(mid_x[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))

    end
    x_edges = x
    x = cat(sub_grids...,dims = dims + 1)



    function eval_function(F,x = x,dims = dims)

        return F([x[[(:) for j in 1:dims]...,i] for i in 1:dims])

    end

    function ip(a,b;weighted = true,omega = omega,dims = dims,combine_channels = true)
        if weighted
            IP = a .* omega .* b
        else
            IP = a .* b
        end
        if combine_channels == true
            IP =  sum(IP,dims = collect(1:(dims+1)))
        else
            IP =  sum(IP,dims = collect(1:(dims)))
        end
        return IP
    end

    function integ(a;weighted = true,omega = omega,dims = dims,ip = ip)
        #channel_a = a[[(:) for i in 1:dims]...,channel:channel,:]
        some_ones = ones(size(a))
        return ip(some_ones,a,weighted=weighted,omega=omega,dims=dims,combine_channels = false)
    end



    return mesh_struct(dims,size(omega),x,x_edges,omega,eval_function,ip,integ,UPC)
end


function gen_permutations(N)

    N_grid = [collect(1:n) for n in N]

    sub_grid = ones(Int,(N...))

    dims = length(N)
    sub_grids = []

    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1


        push!(sub_grids,permutedims(N_grid[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))

    end

    return reshape(cat(sub_grids...,dims = dims + 1),(prod(N)...,dims))
end

function reshape_for_local_SVD(input,MP; subtract_average = false)
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    dims = length(J)



    offsetter = [J...]
    loop_over = gen_permutations(I)
    data = []
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        first_index = offsetter .* (i .-1 ) .+ 1
        second_index = offsetter .* (i)
        index = [(first_index[i]:second_index[i]) for i in 1:dims]
        index = [index...,(:),(:)]
        to_push = input[index...]
        if subtract_average
            to_push .-= mean(to_push,dims = collect(1:dims))
        end
        push!(data,to_push)
    end

    return cat(data...,dims = dims + 2)
end




function carry_out_local_SVD(input,MP;subtract_average = false)
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    reshaped_input = reshape_for_local_SVD(input,MP,subtract_average = subtract_average)

    vector_input = reshape(reshaped_input,(prod(size(reshaped_input)[1:end-1]),size(reshaped_input)[end]))

    SVD = svd(vector_input)
    return reshape(SVD.U,(J...,UPC,Int(size(SVD.U)[end]))),SVD.S
end



function local_to_global_modes(modes,MP)
    number_of_modes = size(modes)[end]
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims


    some_ones = ones(size(modes)[1:end]...,prod(I))
    global_modes = modes .* some_ones

    original_dims = collect(1:length(size(global_modes)))
    permuted_dims = copy(original_dims)
    permuted_dims[end] = original_dims[end-1]
    permuted_dims[end-1] = original_dims[end]

    global_modes = permutedims(global_modes,permuted_dims)

    global_modes = reshape(global_modes,(J...,UPC, I...,number_of_modes))

    output = zeros(I..., J...,UPC,number_of_modes)


    loop_over = gen_permutations((J...,UPC))
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        output[[(:) for j in 1:dims]...,i...,:] = global_modes[i...,[(:) for j in 1:dims]...,:]
    end

    to_reconstruct = reshape(output,(I..., prod(J)*UPC,number_of_modes))

    return reshape(reconstruct_signal(to_reconstruct,J),(([I...] .* [J...])...,UPC,number_of_modes))
end

function compute_overlap_matrix(modes)
    dims = length(size(modes)) -2
    overlap = zeros(size(modes)[end],size(modes)[end])
    for i in 1:size(modes)[end]
        input_1 = modes[[(:) for k in 1:dims+1]...,i:i]
        #input_1 = reshape(input_1,(size(input_1)...,1))
        for j in 1:size(modes)[end]
            input_2 = modes[[(:) for k in 1:dims+1]...,j:j]
            #input_2 = reshape(input_2,(size(input_2)...,1))
            overlap[i,j] = sum(input_1 .* input_2, dims = collect(1:dims+1))[1]
        end
    end
    return overlap
end

function add_filter_to_modes(POD_modes,MP;orthogonalize = false)

    dims = MP.fine_mesh.dims
    UPC = MP.fine_mesh.UPC
    sqrt_omega_tilde = sqrt.(MP.omega_tilde)
    some_zeros = zeros(size(MP.omega_tilde))

    modes = cat([sqrt_omega_tilde,(some_zeros for i in 1:UPC-1)...]...,dims = dims + 1)

    modes = cat([circshift(modes,([0 for i in 1:dims]...,j)) for j in 0:(UPC-1)]...,dims = dims + 2)
    if POD_modes != 0

        modes = cat([modes,POD_modes]...,dims = dims + 2)
    end

    r = size(modes)[dims + 2]
    IP = 0

    for i in 2:r

        s_i = [[(:) for k in 1:dims+1]...,i:i]

        mode_i = modes[s_i...]
        if orthogonalize ### orthogonalize basis using gramm-schmidt
            for j in 1:(i-1)
                s_j = [[(:) for k in 1:dims+1]...,j:j]
                mode_j = modes[s_j...]
                IP = sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_j .* mode_i)),dims = collect(1:dims+1))

                modes[s_i...] .-= (IP) .* mode_j
            end
            mode_i = modes[s_i...]
        end
        norm_i =  sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_i .* mode_i)),dims = collect(1:dims+1))
        modes[s_i...] ./= sqrt.(norm_i)
    end

    return modes
end



struct mesh_pair_struct
    fine_mesh
    coarse_mesh
    J
    I
    one_filter
    one_reconstructor
    omega_tilde
end

struct projection_operators_struct
    Phi_T
    Phi
    W
    R
    r
end





#POD_modes

function gen_projection_operators(POD_modes,MP;uniform = false)

    dims = MP.fine_mesh.dims
    J = MP.J
    I = MP.I

    sqrt_omega_tilde = sqrt.(MP.omega_tilde)
    inv_sqrt_omega_tilde = 1 ./ sqrt_omega_tilde

    if uniform == false

        Phi_T(input,modes = POD_modes,MP = MP) = cat([sum(MP.one_filter(input .* modes[[(:) for i in 1:MP.fine_mesh.dims+1]...,j]),dims = [MP.fine_mesh.dims+1]) for j in 1:size(modes)[end]]...,dims = MP.fine_mesh.dims + 1)

        function Phi(input,modes = POD_modes,MP = MP)
            UPC = MP.fine_mesh.UPC
            dims = MP.fine_mesh.dims
            r = size(modes)[end]
            Phi_mask = ones((size(input)[1:end-2]...,UPC,size(input)[end]))
            result = stop_gradient() do
                zeros((size(modes)[1:dims]...),UPC,size(input)[end])
            end
            for j in 1:r
                result += modes[[(:) for i in 1:dims+1]...,j:j] .* MP.one_reconstructor(input[[(:) for i in 1:dims]...,j:j,:] .* Phi_mask)
            end
            return result
        end
    else
        weights = POD_modes[[(1:J[i]) for i in 1:dims]...,:,:]

        @assert dims <= 1 "Uniform Phi is not supported for dims > 1 at this time, set uniform = false"

        for i in 1:dims
            weights = reverse(weights,dims = i)
        end

        Phi_T = Conv(J, size(weights)[dims+1]=>size(weights)[dims+2],stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
        Phi = ConvTranspose(J, size(weights)[dims+2]=>size(weights)[dims+1],stride = J,pad = 0,bias =false)  # First c

        Phi_T.weight .= weights
        Phi.weight .= weights

    end





    W(input,Phi_T = Phi_T, sqrt_omega_tilde = sqrt_omega_tilde) =  Phi_T(input .* sqrt_omega_tilde)
    R(input,Phi = Phi,inv_sqrt_omega_tilde =inv_sqrt_omega_tilde) =  inv_sqrt_omega_tilde .*  Phi(input)


    return projection_operators_struct(Phi_T,Phi,W,R,r)
end



function generate_coarse_from_fine_mesh(fine_mesh,J)

    divide = [fine_mesh.N...] .% [J...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible reduction parameter J in each dimension."
    end


    dims = fine_mesh.dims
    N = fine_mesh.N
    x = fine_mesh.x_edges

    I =Tuple([Int(fine_mesh.N[i]/J[i]) for i in 1:dims])

    X  = []
    for i in 1:length(x)
        selector = [1,(1 .+ J[i]*collect(1:I[i]))...]
        push!(X,x[i][selector])
    end
    return gen_mesh(X,UPC = fine_mesh.UPC)
end



function find_padding_size(CNN,test_size = 100)
    dims = length(size(CNN[1].weight)) - 2
    input_channels = size(CNN[1].weight)[dims + 1]
    test_input = zeros(Tuple([[test_size for i in 1:dims]...,input_channels,1]))
    test_output = CNN(test_input)
    required_padding = ([size(test_input)...] .- [size(test_output)...])[1:dims]
    return Tuple(Int.(required_padding ./ 2))
end

function conv_NN(widths,channels,strides = 0,bias = true)
    if strides == 0
        strides = ones(Int,size(widths)[1])
    end
    pad = Tuple(zeros(Int,length(widths[1])))
    storage = []
    for i in 1:size(widths)[1]
        kernel_size = Tuple(2* [widths[i]...] .+ 1)
        if i == size(widths)[1]
            storage = [storage;Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,bias = bias)]
        else

            storage = [storage;Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,relu,bias = bias)]
        end
    end
    return Chain((i for i in storage)...)
end





function gen_channel_mask(data,channel)
    dims = length(size(data)) - 2
    number_of_channels = size(data)[end-1]
    channel_mask = stop_gradient() do
        zeros(size(data)[1:end-1])
    end
    stop_gradient() do
        channel_mask[[(:) for i in 1:dims]...,channel] .+= 1
    end
    return channel_mask
end

function construct_corner_mask(N,pad_size)
    dims = length(N)
    corner_mask = zeros(N)
    for i in 1:dims
        original_dims = collect(1:length(size(corner_mask)))
        new_dims = copy(original_dims)
        new_dims[1] = original_dims[i]
        new_dims[i] =   1

        corner_mask = permutedims(corner_mask,new_dims)
        pad_start = corner_mask[(end-pad_size[i]+1):end,[(:) for j in 1:(dims-1)]...]
        pad_end = corner_mask[1:pad_size[i],[(:) for j in 1:(dims-1)]...]




        if i == dims
            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)

            for j in 1:dims-1
                select_start = [(:) for k in 1:(j-1)]
                select_end = [(:) for k in j:(dims-1)]
                pad_start[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_start[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1
                pad_end[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_end[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1

            end

            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)
            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)
        else

            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)

        end
        corner_mask = permutedims(corner_mask,new_dims)

    end
    return corner_mask
end





# figure out how to deal with BCs
# Possibly implement energy conserving auto-encoder
# finish boundary condition indicator (maybe filtered level, maybe ROM level, probably start with filtered)

# connect to NS code
function gen_one_filter(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    filter = Conv(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    filter.weight .= 1.
    return filter
end



function gen_one_reconstructor(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    reconstructor = ConvTranspose(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    reconstructor.weight .= 1.
    return reconstructor
end


function reconstruct_signal(R_q,J)

    ndims(R_q) > 2 || throw(ArgumentError("expected x with at least 3 dimensions"))
    d = ndims(R_q) - 2
    sizein = size(R_q)[1:d]
    cin, n = size(R_q, d+1), size(R_q, d+2)
    #cin % r^d == 0 || throw(ArgumentError("expected channel dimension to be divisible by r^d = $(
    #    r^d), where d=$d is the number of spatial dimensions. Given r=$r, input size(x) = $(size(x))"))

    cout = cin ÷ prod(J)
    R_q = reshape(R_q, sizein..., J..., cout, n)
    perm = hcat(d+1:2d, 1:d) |> transpose |> vec  # = [d+1, 1, d+2, 2, ..., 2d, d]
    R_q = permutedims(R_q, (perm..., 2d+1, 2d+2))
    R_q = reshape(R_q, J.*sizein..., cout, n)

    return R_q
end

function gen_mesh_pair(fine_mesh,coarse_mesh)
    divide = [fine_mesh.N...] .% [coarse_mesh.N...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible by the dimensions of the coarse mesh."
    end
    UPC = fine_mesh.UPC
    dims = fine_mesh.dims
    J =Tuple([Int(fine_mesh.N[i]/coarse_mesh.N[i]) for i in 1:dims])
    I = coarse_mesh.N

    one_filter = gen_one_filter(J,UPC)

    one_reconstructor = gen_one_reconstructor(J,UPC)

    omega_tilde = fine_mesh.omega

    omega_UPC = cat([coarse_mesh.omega for i in 1:UPC]...,dims = dims + 1)
    omega_UPC = reshape(omega_UPC,(size(omega_UPC))...,1)
    omega_tilde = fine_mesh.omega ./ one_reconstructor(omega_UPC)[[(:) for i in 1:dims]...,1,1]


    return mesh_pair_struct(fine_mesh,coarse_mesh,J,I,one_filter,one_reconstructor,omega_tilde)
end


function padding(data,pad_size;circular = false,UPC = 0,BCs = 0,zero_corners = true,navier_stokes = false)


    dims = length(size(data)) - 2
    if navier_stokes == false
        UPC = 0
    elseif UPC == 0
        UPC = dims
    end


    if navier_stokes
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims,UPC)
            end
        end
    else
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims)
            end
        end
    end



    N = size(data)[1:dims]
    if navier_stokes && (circular == false)
        split_data = [data[[(:) for i in 1:dims]...,j:j,:] for j in 1:UPC]
        unknown_index = 0
        padded_data = []
        for data in split_data
            unknown_index += 1
            for i in 1:dims
                original_dims = stop_gradient() do
                    collect(1:length(size(data)))
                end
                new_dims = stop_gradient() do
                    copy(original_dims)
                end
                stop_gradient() do
                        new_dims[1] = original_dims[i]
                        new_dims[i] = 1
                end

                data = permutedims(data,new_dims)

                pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]


                if circular == false
                    #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                    if BCs[1,i,unknown_index] != "c" && BCs[1,i,unknown_index] != "m"
                        pad_start_cache = 2* BCs[1,i,unknown_index] .- reverse(pad_end,dims = 1)

                    elseif BCs[1,i,unknown_index] == "m"
                        pad_start_cache = reverse(pad_end,dims = 1)
                    else
                        pad_start_cache = pad_start
                    end

                    if BCs[2,i,unknown_index] != "c" && BCs[2,i,unknown_index] != "m"
                        pad_end_cache = 2* BCs[2,i,unknown_index] .- reverse(pad_start,dims = 1)
                    elseif BCs[2,i,unknown_index] == "m"
                        pad_end_cache = reverse(pad_start,dims = 1)
                    else
                        pad_end_cache = pad_end
                    end


                    pad_start = pad_start_cache
                    pad_end = pad_end_cache
                end

                data = cat([pad_start,data,pad_end]...,dims = 1)
                data = permutedims(data,new_dims)

            end
            push!(padded_data,data)
        end

        padded_data = cat(padded_data...,dims = dims + 1)

        if size(data)[dims+1] > UPC
            data = data[[(:) for i in 1:dims]...,UPC+1:end,:]
            for i in 1:dims
                original_dims = stop_gradient() do
                    collect(1:length(size(data)))
                end
                new_dims = stop_gradient() do
                    copy(original_dims)
                end
                stop_gradient() do
                        new_dims[1] = original_dims[i]
                        new_dims[i] = 1
                end

                data = permutedims(data,new_dims)

                pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]


                if circular == false
                    #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                    BC_right = BCs[1,i,:]
                    BC_left = BCs[2,i,:]


                    if BC_right[1] != "c"
                        for j in BC_right
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        BC_right = 2*(BC_right .== "m") .- 1
                        BC_left = 2*(BC_left .== "m") .- 1

                        pad_start_cache = reverse(pad_end,dims = 1)
                        pad_end_cache = reverse(pad_start,dims = 1)
                        #if mirror_mask != 0
                        #    pad_start_cache = multiply_by_mirror_mask(pad_start_cache,mirror_mask[i][BC_right],UPC)
                        #    pad_end_cache = multiply_by_mirror_mask(pad_end_cache,mirror_mask[i][BC_left],UPC)
                        #end
                        pad_start = pad_start_cache
                        pad_end = pad_end_cache
                    else
                        for j in BC_right
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end


                    end

                    pad_start


                end

                data = cat([pad_start,data,pad_end]...,dims = 1)
                data = permutedims(data,new_dims)

            end
            padded_data = cat(padded_data,data,dims = dims + 1)
        end

    else
        for i in 1:dims

            original_dims = stop_gradient() do
                collect(1:length(size(data)))
            end
            new_dims = stop_gradient() do
                copy(original_dims)
            end
            stop_gradient() do
                    new_dims[1] = original_dims[i]
                    new_dims[i] = 1
            end

            data = permutedims(data,new_dims)

            pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
            pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]


            if circular == false

                #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                if BCs[1,i,1] != "c" && BCs[1,i,1] != "m"

                    pad_start_cache = BCs[1,i,1] .* pad_start.^0
                elseif BCs[1,i,1] == "m"
                    pad_start_cache = reverse(pad_end,dims = 1)
                else
                    pad_start_cache = pad_start
                end

                if BCs[2,i,1] != "c" && BCs[2,i,1] != "m"
                    pad_end_cache = BCs[2,i,1] .* pad_end.^0
                elseif BCs[2,i,1] == "m"
                    pad_end_cache = reverse(pad_start,dims = 1)
                else
                    pad_end_cache = pad_end
                end


                pad_start = pad_start_cache
                pad_end = pad_end_cache
            end

            data = cat([pad_start,data,pad_end]...,dims = 1)
            data = permutedims(data,new_dims)
        end
        padded_data = data
    end




    if zero_corners == true && circular == false
        corner_mask = stop_gradient() do
            construct_corner_mask(N,pad_size)
        end
        padded_data = padded_data .- corner_mask .* padded_data
    end

    return  padded_data
end






###################### Neural network code ############

# What to do with padding
# What to do with multiple unknowns, i.e. u and v field
# How to save the neural network

struct model_struct
    eval
    CNN
    r
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end





########### Time integration code ####

#function rhs(input,mesh,t;other_arguments = (model,))
    # implement boundary conditions through clever use of the "other_arguments" input
#    return other_arguments[1].eval(input)
#end

function time_step(input,mesh,t,dt,rhs;other_arguments = 0,method = "RK4")
    if method == "RK4"

        k1 = rhs(input,mesh,t,other_arguments = other_arguments)
        k2 = rhs(input .+ dt*k1/2,mesh,t .+ dt/2,other_arguments = other_arguments)
        k3 = rhs(input .+ dt*k2/2,mesh,t .+ dt/2,other_arguments = other_arguments)
        k4 = rhs(input .+ dt*k3,mesh,t .+ dt,other_arguments = other_arguments)

        return 1/6*dt*(k1 .+ 2*k2 .+ 2*k3 .+ k4)
    end
end


function simulate(input0,mesh,dt,t_start,t_end,rhs,time_step_function;save_every = 1,other_arguments = 0,pre_allocate = false)

    dims = length(size(input0))-2


    steps = stop_gradient() do
       round(Int,(t_end[1] - t_start[1]) ./ dt)
    end

    if pre_allocate == false
        output = stop_gradient() do
            Array{Float64}(undef, size(input0)[1:end]..., 0)
        end

        output_t = stop_gradient() do
            Array{Float64}(undef, ones(Int,dims + 1)...,size(input0)[end], 0)
        end
    else
        output = stop_gradient() do
            zeros(size(input0)[1:end]..., floor(Int,steps/save_every))
        end

        output_t = stop_gradient() do
            zeros(ones(Int,dims + 1)...,size(input0)[end], floor(Int,steps/save_every))
        end
    end

    if length(size(t_start)) == 0
        t_start = stop_gradient() do
            [t_start]
        end
    end


    t = stop_gradient() do
        ones(size(output_t)[1:end-2]...,size(input0)[end]) .* reshape(t_start,(size(output_t)[1:end-2]...,prod(size(t_start))))
    end
    input = input0
    pre_alloc_counter = 0
    for i in 1:steps
        input += time_step_function(input,mesh,t,dt,rhs;other_arguments = other_arguments)
        t  = t .+ dt
        if i % save_every == 0
            pre_alloc_counter += 1
            if pre_allocate == false
                output = cat([output,input]...,dims = dims + 3)
                output_t = cat([output_t,t]...,dims = dims + 3)
            else
                output[[(:) for i in size(input)]...,pre_alloc_counter] += input
                output_t[[(:) for i in size(t)]...,pre_alloc_counter] += t
            end
        end
    end

    return output_t,output
end

function gen_time_interpolator(t_data,data) # only for uniform timesteps
        # supply t_data as e.g. (1,1,1,number_of_simulations,number_of_time_steps) and
        # data as (N,N,UPC,number_of_simulations,number_of_time_steps) sized array
    function interpolator_function(t;simulation_indexes = (:),data = data,t_data = t_data)
        # supply t as (1,1,1,considered_number_of_simulations,considered_points_in_time) and
        # simulation indexes as a (considered_number_of_simulations) sized array
        dims = length(size(data))-3

        data = data[[(:) for i in 1:dims+1]...,simulation_indexes,:]
        t_data = t_data[[(:) for i in 1:dims+1]...,simulation_indexes,:]

        t_start = t_data[[(:) for i in 1:dims+1]...,:,1:1]
        t_end =  t_data[[(:) for i in 1:dims+1]...,:,end:end]
        number_of_time_steps = size(t_data)[end] .- 1


        indexes = number_of_time_steps .* (((t .* ones(size(t_start))) .- t_start) ./ (t_end .- t_start)) .+ 1
        lower_index = floor.(Int,indexes)
        higher_index = ceil.(Int,indexes)
        weight = indexes - lower_index

        lower_data = cat([data[[(:) for i in 1:dims+1]...,j,lower_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(data)[dims+2]]...,dims = dims +2)
        higher_data = cat([data[[(:) for i in 1:dims+1]...,j,higher_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(data)[dims+2]]...,dims = dims +2)

        #lower_data = cat([t_data[[(:) for i in 1:dims+1]...,j,lower_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(t_data)[dims+2]]...,dims = dims +2)
        #higher_data = cat([t_data[[(:) for i in 1:dims+1]...,j,higher_index[[1 for i in 1:dims+1]...,j:j,:]] for j in 1:size(t_data)[dims+2]]...,dims = dims +2)



        interpolated = weight .* higher_data + (1 .- weight) .* lower_data
        return interpolated
    end
end


############### Code for skew symmetric neural network #######################

using NNlib

function cons_mom_B(B_kernel;channel = 1)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        channel_mask = gen_channel_mask(B_kernel,channel)

        means = mean(B_kernel,dims = collect(1:dims))
        return B_kernel .- means .* channel_mask
    else
        return 0
    end
end

function transpose_B(B_kernel)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        original_dims = stop_gradient() do
           collect(1:dims+2)
        end
        permuted_dims = stop_gradient() do
           copy(original_dims)
        end

        stop_gradient() do
            permuted_dims[dims+1] = original_dims[dims+2]
            permuted_dims[dims+2] = original_dims[dims+1]
        end

        T_B_kernel = permutedims(B_kernel,permuted_dims)

        for i in 1:dims
           T_B_kernel = reverse(T_B_kernel,dims = i)

        end

        return T_B_kernel
    else
        return 0
    end
end


struct skew_model_struct
    eval
    CNN
    r
    B
    B_mats
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end

function gen_skew_NN(kernel_sizes,channels,strides,r,B;UPC = 0,boundary_padding = 0,constrain_energy = true,conserve_momentum=true,dissipation = true)
    if boundary_padding != 0 && boundary_padding != "c"
        add_input_channel = zeros(Int,size(channels)[1]+1)
        add_input_channel[1] += 1
    else
        add_input_channel = 0
    end

    if dissipation && constrain_energy
       channels = [channels ; 2*r]
    else
       channels = [channels ; r]
    end
    CNN = conv_NN(kernel_sizes,channels .+ add_input_channel,strides)
    pad_size = find_padding_size(CNN)

    if UPC == 0
        UPC = length(size(model.CNN[1].weight))-2
    end
    dims = length(size(CNN[1].weight))-2


    B1,B2,B3 = 0,0,0
    if constrain_energy
        B1 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        B2 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        if dissipation
            B3 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        end
    end


    if constrain_energy
        pad_size = [pad_size...]
        pad_size .+= [B...]
        pad_size = Tuple(pad_size)
    end


    B_mats = [B1,B2,B3]


    function NN(input;a = 0,CNN = CNN,r = r,B = [B...],B_mats = B_mats,UPC = UPC,pad_size = pad_size,boundary_padding = boundary_padding,constrain_energy =constrain_energy,conserve_momentum = conserve_momentum,dissipation = dissipation)

        dims = length(size(input)) - 2
        #CNN[1].weight .*= 4

        #a = input[[(:) for i in 1:dims]...,1:r,:]
        #if constrain_energy
        #    input = input[[(2*B[i]+1:end - 2*B[i]) for i in dims]...,:,:]
        #end
        ### deal with BCs in the CNN #######
        ####################################

        if boundary_padding == 0 || boundary_padding == "c"
            output = CNN(padding(input,pad_size,circular = true))

        else
            pad_input = padding(input,pad_size,BCs = boundary_padding)
            boundary_indicator_channel = stop_gradient() do
                ones(size(input)[1:end-2]...,1,size(input)[end])
            end
            boundary_indicator_padding = stop_gradient() do
                copy(boundary_padding)
            end
            stop_gradient() do
                for i in 1:prod(size(boundary_indicator_padding))
                    if boundary_indicator_padding[i] != "c"
                        boundary_indicator_padding[i] = i + 1
                    end
                end
            end
            pad_boundary_indicator_channel = padding(boundary_indicator_channel,pad_size,BCs = boundary_indicator_padding)
            output = CNN(cat([pad_input,pad_boundary_indicator_channel]...,dims = dims + 1))
        end
        #############################
        ##############################

        phi = output[[(:) for i in 1:dims]...,1:r,:]


        psi = 0
        if constrain_energy && dissipation
            psi = output[[(:) for i in 1:dims]...,r+1:2*r,:]
            #dTd = sum( d.^2 ,dims = [i for i in 1:dims])
        else
            psi = 0
        end


        B1,B2,B3 = B_mats



        if conserve_momentum && constrain_energy
            B1 = cons_mom_B(B1)
            B2 = cons_mom_B(B2)
            B3 = cons_mom_B(B3)
        end
        B1_T,B2_T,B3_T = 0,0,0
        if constrain_energy

            B1_T = transpose_B(B1)

            B2_T = transpose_B(B2)
            B3_T = transpose_B(B3)
        else
            B1_T,B2_T,B3_T = 0,0,0
        end

        c_tilde = 0
        if constrain_energy # skew_symmetric_form
            c_tilde = NNlib.conv(NNlib.conv(a,B1) .* phi,B2_T) - NNlib.conv(NNlib.conv(a,B2) .* phi,B1_T)
            if dissipation
                c_tilde -=  NNlib.conv(psi.^2 .* NNlib.conv(a,B3),B3_T)
            end
        else

            c_tilde = phi

        end

        #c_tilde = phi
        return  c_tilde
    end



    return skew_model_struct(NN,CNN,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides)
end


function save_skew_model(model,name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    mkpath(name)
    save(name * "/model_state.jld","CNN_weights_and_biases",[(i.weight,i.bias) for i in model.CNN],"r",model.r,"B",model.B,"B_mats",model.B_mats,"UPC",model.UPC,"pad_size",model.pad_size,"boundary_padding",model.boundary_padding,"constrain_energy",model.constrain_energy,"conserve_momentum",model.conserve_momentum,"dissipation",model.dissipation,"kernel_sizes",model.kernel_sizes,"channels",model.channels,"strides",model.strides)
    print("\nModel saved at directory [" * name * "]\n")
end

function load_skew_model(name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    CNN_weights_and_biases,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides = (load(name * "/model_state.jld")[i] for i in ("CNN_weights_and_biases","r","B","B_mats","UPC","pad_size","boundary_padding","constrain_energy","conserve_momentum","dissipation","kernel_sizes","channels","strides"))

    model = gen_skew_NN(kernel_sizes,channels,strides,r,B,boundary_padding = boundary_padding,UPC = coarse_mesh.UPC,constrain_energy = constrain_energy,dissipation = dissipation,conserve_momentum = conserve_momentum)

    for i in 1:length(model.CNN)
        model.CNN[i].weight .= CNN_weights_and_biases[i][1]
        model.CNN[i].bias .= CNN_weights_and_biases[i][2]
    end
    for i in 1:length(model.B_mats)
        model.B_mats[i] = B_mats[i]
    end

    print("\nModel loaded from directory [" * name * "]\n")
    return model
end
