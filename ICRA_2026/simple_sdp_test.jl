using LinearAlgebra 
using Convex
using SCS 

# ------------------------------- Problem setup ------------------------------ #
#problem parameters
N = 31 #number of timesteps
x_initial = [-10; 0.1; 0; 0]
x_obs = [-5.0, 0.0]  #position of the obstacle center
r_obs = 2 #radius of the obstacle 

q_xx = 0.1
r_xx = 10.0
R_xx = 500.0
reg = 1e-6

# ------------------------------ System dynamics ----------------------------- #
#dimensions
nx = 4  # state dimension x = [position, velocity]
nu = 2 # controls dimension u = [acceleration]

nxx = 16 #number of elements in xx'
nxu = 8 #number of elements in ux'
nux = 8 #number of elements in xu'
nuu = 4 #number of elements in uu'

Ad = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]
Bd = [0.5 0; 0 0.5; 1 0; 0 1]

A = [Ad zeros(nx, nxx); 
     zeros(nxx, nx) kron(Ad, Ad)]

B = [Bd zeros(nx, nxu + nux + nuu); 
     zeros(nxx, nu) kron(Bd, Ad) kron(Ad, Bd) kron(Bd, Bd)]

# ------------------------- SDP generation and solve ------------------------- #
#weights 
Q = zeros(nx + nxx, nx + nxx) + reg*Matrix(I, nx + nxx, nx + nxx)
q = [zeros(nx); vec(q_xx*Matrix(I, nx, nx))]

R = Diagonal([zeros(nu); zeros(nxu+nux); vec(R_xx*Matrix(I, nu, nu))])+reg*Matrix(I, nu + nxu + nux + nuu, nu + nxu + nux + nuu)
r = [zeros(nu + nxu + nux); vec(r_xx*Matrix(I, nu, nu))]

#decision variables
x_bar = Variable(nx + nxx, N)
u_bar = Variable(nu + nxu + nux + nuu, N-1) 

global obj = 0
constraints = []
for k=1:N

    # initial condition
    if k == 1
        push!(constraints, x_bar[:,1] == [x_initial; vec(x_initial*x_initial')])
    end

    #dynamics constraints 
    if k < N
        push!(constraints, x_bar[:,k+1] == A*x_bar[:,k] + B*u_bar[:,k])
    end

    #PSD constraints 
    x = x_bar[1:nx,k]
    XX = reshape(x_bar[nx+1:end,k], nx, nx)
    if k < N
        u = u_bar[1:nu,k]
        XU =  reshape(u_bar[nu+1: nu + nxu, k], nx, nu)
        UX = reshape(u_bar[nu+nxu + 1: nu + nxu + nux, k], nu, nx)
        UU = reshape(u_bar[nu+nxu+nux+1:end, k], nu, nu)
        push!(constraints, [1 x' u';
                            x XX XU;
                            u UX UU]⪰ 0)
    else
        push!(constraints, [1 x'; 
                            x XX] ⪰ 0)
    end

    # collision avoidance
    push!(constraints, tr(XX[1:2, 1:2])- 2*x_obs'*x[1:2] + x_obs'*x_obs - r_obs^2 >= 0)

    # cost function
    global obj += quadform(x_bar[:,k], Q) + q'*x_bar[:,k]
    if k < N
        global obj += quadform(u_bar[:,k], R) + r'*u_bar[:,k] 
    end

end

#solve problem
problem = minimize(obj, constraints)
println("Solving SDP problem with ", length(constraints), " constraints...")

# Solve with SCS solver
solve!(problem, SCS.Optimizer)
println("Problem Status: ", problem.status)

if problem.status == Convex.OPTIMAL
    println("✅ SDP solution found!")
    
    #extract solution
    x_bar_opt = x_bar.value
    u_bar_opt = u_bar.value 
    x_opt = x_bar_opt[1:4,:]
    u_opt = u_bar_opt[1:2,:]
    
    # Save trajectory data for comparison
    open("julia_sdp_trajectory.csv", "w") do io
        println(io, "# Julia SDP Reference Solution")
        println(io, "# Format: time, pos_x, pos_y, vel_x, vel_y, u_x, u_y")
        
        for k in 1:N
            print(io, k-1, ", ", x_opt[1,k], ", ", x_opt[2,k], ", ", x_opt[3,k], ", ", x_opt[4,k])
            if k < N
                println(io, ", ", u_opt[1,k], ", ", u_opt[2,k])
            else
                println(io, ", 0, 0")
            end
        end
    end
    
    println("📊 Trajectory saved to julia_sdp_trajectory.csv")
    
    # Safety check
    violations = 0
    min_dist = Inf
    for k in 1:N
        pos = x_opt[1:2, k]
        dist = norm(pos - x_obs)
        min_dist = min(min_dist, dist)
        if dist < r_obs
            violations += 1
        end
    end
    
    println("🛡️ Safety Analysis:")
    println("   Violations: ", violations, "/", N)
    println("   Min distance: ", round(min_dist, digits=3))
    println("   Safe: ", violations == 0 ? "YES" : "NO")
    
    # Check constraint residuals
    check = zeros(N)
    for i=1:N 
        check[i] = sum(x_opt[:,i]*x_opt[:,i]' - reshape(x_bar_opt[nx+1:end, i], nx,nx))
    end
    max_residual = maximum(abs.(check))
    println("   Max PSD residual: ", round(max_residual, digits=6))
    
    println("✅ Julia SDP reference solution complete!")
    
else
    println("❌ SDP solve failed with status: ", problem.status)
end
