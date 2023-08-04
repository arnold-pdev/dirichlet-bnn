# plot contour of Dirichlet distribution for 3 variables 
using Plots, Distributions, CairoMakie, Observables

function dirichletplot(α)
    x = 0.0:0.001:1.0
    y = sqrt(3)/2*x

    # affine projection of triangle to 2D plane
    L(x,y) = [ 1 -1/sqrt(3); 0 2/sqrt(3); -1 -1/sqrt(3) ]*[x,y] + [0,0,1]
    p(x,y) = pdf(Dirichlet(α), L(x,y))
    z = @. p(x', y)

    contourf(x, y, z, levels = 100*collect(0.0:0.05:1.0), linewidth = 0)

    # contourf(x, y, z, levels = maximum(filter(isfinite, z))*(0.0:0.1:1.0))
    plot!([0, 1, 0.5, 0], [0, 0, sqrt(3)/2, 0], color = :white, legend = false)
    # scatter!()
end

function dirichletanim(A)
    x = 0.0:0.001:1.0
    y = sqrt(3)/2*x
    α = Observable(A[1])

    # affine projection of triangle to 2D plane
    L(x,y) = [ 1 -1/sqrt(3); 0 2/sqrt(3); -1 -1/sqrt(3) ]*[x,y] + [0,0,1]
    X = L.(x',y)
    z = lift(α-> pdf(Dirichlet(α), X)', α)

    fig = Figure(resolution = (600, 600), fontsize = 24)
    ax1 = Axis(fig[1,1], title = "Dirichlet distribution", aspect = 1, backgroundcolor = :transparent)
        CairoMakie.contourf!(ax1, x, y, z, levels = 100*collect(0.0:0.05:1.0), lw = 0, colormap = :inferno)

    ax2 = Axis(fig[1,1], aspect = 1, backgroundcolor = :transparent)
        CairoMakie.lines!(ax2, [0, 1, 0.5, 0], [0, 0, sqrt(3)/2, 0], color = :white, legend = false)
        hidedecorations!(ax2)
        hidespines!(ax2)
    function animstep!(i)
        α[] = A[i]
    end

    CairoMakie.record(fig, "dirichlet.mp4", 1:length(A), framerate = 20) do i # L is the number of frames
        animstep!(i)
    end
end

function dirichletanime(A, py=[])
    x = 0.0:0.001:1.0
    y = sqrt(3)/2*x
    α = Observable(A[1])
    data1 = Observable(Vector{Float64}[])
    data2 = Observable(Vector{Float64}[])

    # datax = 

    # affine projection of triangle to 2D plane
    L(x,y) = [ 1 -1/sqrt(3); 0 2/sqrt(3); -1 -1/sqrt(3) ]*[x,y] + [0,0,1]
    X = L.(x',y)
    z = lift(α-> pdf(Dirichlet(α), X)', α)

    fig = Figure(resolution = (600, 600), fontsize = 24)
    ax1 = Axis(fig[1,1], title = "Dirichlet distribution", aspect = 1, backgroundcolor = :transparent)
        CairoMakie.contourf!(ax1, x, y, z, levels = 100*collect(0.0:0.05:1.0), lw = 0, color = :inferno)
        CairoMakie.scatter!(ax1,marker='x', datax, datay, color = :white, markersize = 10)
    ax2 = Axis(fig[1,1], aspect = 1, backgroundcolor = :transparent)
        CairoMakie.lines!(ax2, [0, 1, 0.5, 0], [0, 0, sqrt(3)/2, 0], color = :white, legend = false)
        hidedecorations!(ax2)
        hidespines!(ax2)
    function animstep!(i)
        α[] = A[i]
        data2[] = data[1]
        data1[] = py[i]
    end

    CairoMakie.record(fig, "dirichlet.mp4", 1:length(A), framerate = 20) do i # L is the number of frames
        animstep!(i)
    end
end
