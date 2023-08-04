comp = 7 # component number

Z = @. rand(Dirichlet(α(data_x)),100)
μ = @. α(data_x)/sum(α(data_x))
μc = [ μ[i][comp] for i in eachindex(μ)]
μcr =  [ data_y[i][comp] for i in eachindex(data_y) ]

x = 1917:2557 # validation set
Plots.plot(x, 1.25*ones(length(x)), fill = (0, 0.5, :gray80))
Plots.plot!(x, -0.5*ones(length(x)), fill = (0, 0.5, :gray80))
for i in eachindex(μc)
    # Plots.plot!([i,i],μ1[i] .+ 2*std(Z[i][1,:])*[-1,1],color=:red)
    Plots.plot!([i,i],quantile(Z[i][comp,:],[.05,.95]),color=:green)
end
Plots.scatter!(μc, color=:green)
Plots.scatter!(μcr, color = :black, legend = false, ylim = [0,1])