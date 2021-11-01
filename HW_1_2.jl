#############################
### Numerical Integration ###
#############################

###Script computes utility evaluated over a finite horizon
###using midpoint, trapozoid, and Simpson's quadrature, as
###well as stratified Monte Carlo sampling; returns run time
###for each method###

function main()


T=100
λ=0.02
ρ=0.04
N=1000000


###Utility Function###

function utility(c)

    return -exp(-c)

end

###Define Integrand###

function integrand(λ,ρ,utility,t)

    exp(-ρ*t)*utility(1-exp(-λ*t))

end


###Midpoint method###
function midpoint(ρ,λ,T,N,integrand)
    tic=@elapsed begin
    
    h=T/N
    integral=0
    for i in 1:N
        t=(i-(1/2))*h
        integral=integral+h*integrand(λ,ρ,utility,t)
    end
    end

    return [integral tic]
end

###Trapezoid method####

function trapezoid(ρ,λ,T,N,integrand)
    tic=@elapsed begin

    h=T/N
    integral=integrand(λ,ρ,utility,0)/2
    for i in 1:(N-1)
        x=i*h
       integral=integral+integrand(λ,ρ,utility,x)
    end
    integral=integral+integrand(λ,ρ,utility,T)/2
    integral=integral*h
    end

    return [integral tic]

end

###Simpson###
function simpson(ρ,λ,T,N,integrand)
    tic=@elapsed begin
    
    h=T/N

    ###Add the irregular summands###
    integral=integrand(λ,ρ,utility,0)+5*integrand(λ,ρ,utility,h)
    integral=integral+5*integrand(λ,ρ,utility,T-h)+integrand(λ,ρ,utility,T)

    ###Add the regular summands###
    for i=2:(N-2)
        t=i*h
        integral=integral+integrand(λ,ρ,utility,t)
    end

    integral=integral*h
    end

    return [integral tic]

end

M=10
###Monte Carlo-M strata with N sample points per stratum###
function monteCarloStrat(ρ,λ,T,N,M,integrand)
    tic=@elapsed begin

    α=T/M
    sample=zeros(M,N)
    for i=1:M
        sample[i,:]=α*(i-1)*ones(1,N)+10*rand(Float64,(1,N))
    end
    rangeSample=map((x)->integrand(λ,ρ,utility,x),sample)
    integral=(α/N)*sum(rangeSample)
    end

    return [integral tic]

end
###Midpoint-Trapazoid-Simpson-MonteCarloStrat###
Nmonte=Int(N/M)
println(Nmonte)
return tabular(vcat(midpoint(ρ,λ,T,N,integrand),trapezoid(ρ,λ,T,N,integrand),
simpson(ρ,λ,T,N,integrand),monteCarloStrat(ρ,λ,T,Nmonte,M,integrand)))
end