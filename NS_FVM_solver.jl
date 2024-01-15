include("source.jl")
using FFTW

function gen_A_c_s(mesh)
    dims = mesh.dims
    omega = mesh.omega
    dx = mesh.dx
    UPC = mesh.UPC
    select = [(:);[2 for i in 1:dims-1]]

    stenc_3 = zeros(([3 for i in 1:dims]...))

    A = Conv(([3 for i in 1:dims]...,), UPC=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)
            if i == j
                stencil[circshift(select,(i-1,))...] .= [0,1/2,1/2]
            end
            A.weight[[(:) for k in 1:dims]...,i,j] .= stencil
        end
    end
    return A
end

function gen_A_s_c(mesh)
    dims = mesh.dims
    omega = mesh.omega
    dx = mesh.dx
    UPC = mesh.UPC

    select = [(:);[2 for i in 1:dims-1]]
    stenc_3 = zeros(([3 for i in 1:dims]...))

    A = Conv(([3 for i in 1:dims]...,), UPC=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)
            if i == j
                stencil[circshift(select,(i-1,))...] .= [1/2,1/2,0]
            end
            A.weight[[(:) for k in 1:dims]...,i,j] .= stencil
        end
    end
    return A
end

struct grid_swapper
    mesh
    A_c_s
    A_s_c
    omega_s
    dx_s
    face_area
end

function gen_grid_swapper(mesh)

    omega = mesh.omega
    dx = mesh.dx

    pad_size = ([1 for i in 1:mesh.dims]...,)

    A_c_s_unweighted = gen_A_c_s(mesh)
    A_c_s(V,A_c_s_unweighted = A_c_s_unweighted ,pad_size = pad_size) = A_c_s_unweighted(padding(V,pad_size,circular = true))




    omega_s = A_c_s(omega)
    dx_s =   A_c_s(dx)
    face_area = omega_s ./ dx_s

    A_s_c_unweighted = gen_A_s_c(mesh)

    A_s_c(V,A_s_c_unweighted = A_s_c_unweighted,omega_s = omega_s,omega = omega,pad_size = pad_size) = (1 ./ omega) .* A_s_c_unweighted(padding(omega_s .* V,pad_size;circular = true))


    return grid_swapper(mesh,A_c_s,A_s_c,omega_s,dx_s,face_area)
end


function gen_pressure_solver(mesh,O)
    N = mesh.N
    dims = mesh.dims

    # Find eigenvalues ###
    ##############################

    N = mesh.N
    dims = mesh.dims


    test_hat = ones(N...,1,1)

    test = ifft(test_hat,collect(1:dims))

    pad_test = padding(test,Tuple((1 for i in 1:dims)),circular = true)

    G_test = O.G(pad_test)

    pad_G_test = padding(G_test,Tuple((1 for i in 1:dims)),circular = true)

    M_test = O.M(pad_G_test)

    lambda = real(fft(M_test,collect(1:dims)))

    ############################################
    ############################################

    lambda[[1 for i in 1:dims]...] = 1


    zero_mask = ones(N)


    zero_mask[[1 for i in 1:dims]...] = 0



    function pressure_poisson(r;lambda = lambda,zero_mask = zero_mask)

        r_hat = fft(r,collect(1:dims))

        p_hat =  zero_mask .* (r_hat ./ lambda)


        p =  real.(ifft(p_hat,collect(1:dims)))


        return p
    end

    return pressure_poisson
end


struct operators_struct
    M
    G
    C
    D
    w
end


function gen_operators_uniform(mesh;precision = f32)
    dims = mesh.dims
    h = mesh.dx[1]
    UPC = mesh.UPC

    @assert dims != 3 "Convection for dims = 3 is not yet supported"

    stenc_3 = zeros(([3 for i in 1:dims]...))
    select = [(:);[2 for i in 1:dims-1]]

    M = Conv(([3 for i in 1:dims]...,), UPC=>1,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    M = precision(M)
    for i in 1:UPC
        stencil = copy(stenc_3)

        stencil[circshift(select,(i-1,))...] .= 1/h * [1,-1,0]
        M.weight[[(:) for k in 1:dims]...,i,1] .= stencil
    end

    G = Conv(([3 for i in 1:dims]...,), 1=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    G = precision(G)
    for i in 1:UPC

        stencil = copy(stenc_3)

        stencil[circshift(select,(i-1,))...] .= 1/h * [0,-1,1]

        G.weight[[(:) for k in 1:dims]...,1,i] .= stencil

    end
    pad_size = ([1 for i in 1:mesh.dims]...,)


    A_unshifted = Conv(([3 for i in 1:dims]...,), UPC=>UPC^2,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    A_unshifted = f32(A_unshifted)
    A_unshifted.weight .= 0

    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)

            stencil[circshift(select,(i-1,))...] += [1/2,1/2,0]

            A_unshifted.weight[[(:) for k in 1:dims]...,j,i + (j-1)*UPC] .= stencil
        end
    end

    A_shifted = Conv(([3 for i in 1:dims]...,), UPC=>UPC^2,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    A_shifted = precision(A_shifted)
    A_shifted.weight .= 0
    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)
            if i == j
                stencil[circshift(select,(i-1,))...] += [1/2,1/2,0]

            else ### shift
                shifted_select = [(:);[1 for i in 1:dims-1]]
                stencil[circshift(shifted_select,(i-1,))...] += [0,1/2,1/2]
            end
            A_shifted.weight[[(:) for k in 1:dims]...,j,i + (j-1)*UPC] .= stencil
        end
    end

    nabla = Conv(([3 for i in 1:dims]...,), UPC^2=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    nabla = precision(nabla)
    nabla.weight .= 0

    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)

            stencil[circshift(select,(i-1,))...] += (1/h) .* [0,1,-1]
            nabla.weight[[(:) for k in 1:dims]...,i + (j-1)*UPC,j] .= stencil
        end
    end


    function C(V,h = h,A_unshifted = A_unshifted,A_shifted = A_shifted,dims = dims,nabla = nabla)
        avgs_unshifted = A_unshifted(V)
        avgs_shifted = A_shifted(V)
        reordering = stop_gradient() do
            (reshape(collect(1:dims .^ 2),(dims,dims))')[1:end]
        end
        multiply = avgs_unshifted  .* avgs_shifted[[(:) for i in 1:dims]...,reordering,:]

        return  nabla(multiply)
    end




    if dims == 2
        w = Conv(([3 for i in 1:dims]...,), UPC=>1,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
        w = precision(w)
        for i in 1:UPC
            stencil = copy(stenc_3)
            stencil[circshift(select,(i,))...] .= 1/h * [1,-1,0]
            if i == 1
                w.weight[[(:) for k in 1:dims]...,i,1] .= -stencil
            else
                w.weight[[(:) for k in 1:dims]...,i,1] .= stencil
            end
        end
    else
       w(V) = 0 * V
    end

    function old_C(V,h = mesh.dx[1],dims = mesh.dims)
        V = V[[(3:end-2) for i in 1:dims]...,:,:]
        V_avg_x = 1/2 * (V + circshift(V,(-1,0,0,0)))
        V_avg_y = 1/2 * (V + circshift(V,(0,-1,0,0)))

        u_xx = V_avg_x[:,:,1:1,:] .* V_avg_x[:,:,1:1,:]
        u_xy = circshift(V_avg_x[:,:,2:2,:],(1,-1,0,0)) .* V_avg_y[:,:,1:1,:]

        v_yy = V_avg_y[:,:,2:2,:] .* V_avg_y[:,:,2:2,:]
        v_yx = circshift(V_avg_y[:,:,1:1,:],(-1,1,0,0)) .* V_avg_x[:,:,2:2,:]


        C_x = (u_xx - circshift(u_xx,(1,0,0,0)) + (u_xy - circshift(u_xy,(0,1,0,0))))
        C_y = (v_yy - circshift(v_yy,(0,1,0,0)) + (v_yx - circshift(v_yx,(1,0,0,0))))

        return (1/h)*cat(C_x,C_y,dims = dims + 1)
    end

    D = Conv(([3 for i in 1:dims]...,), UPC=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    D = precision(D)
    for i in 1:UPC
        for j in 1:UPC
            stencil = copy(stenc_3)
            if i == j
                for k in 1:dims
                    stencil[circshift(select,(k-1,))...] += (1/(h^2)) * [1,-2,1]
                end
            end
            D.weight[[(:) for k in 1:dims]...,i,j] .= stencil
        end
    end


    return operators_struct(M,G,C,D,w)#,Q,Q_T,D
end
