##########################################
###Social planners maximization problem###
##########################################

###Script solves static social planner problem with
###n households, m goods, and household heterogeneity
###in preferences###

###Define objective function parameters###



using LinearAlgebra
using LatexPrint

n=3
m=3

α=ones(m) ##multiplicative parameters
α[1]=4
ω=-1.5*ones(n,m) ##CRRA parameters
for i=1:n
    ω[i,i]=-3
end
λ=ones(1,n) ##Pareto weights
λ[1]=3
e=ones(1,m)

#################################################################
###Defining social planner's objective and auxillary functions###
#################################################################

function objective(x,α,ω,λ,e)

    n=size(x)[1]+1
    m=size(x)[2]
    utility=zeros(n,1)
    for i=1:(n-1)
        for j=1:m
            utility[i]=α[j]*(x[i,j]^(ω[i,j]-1))/(ω[i,j]-1)+utility[i]
        end
        utility[i]=λ[i]*utility[i]
    end

    ###Remainder for last person (since there are n-1 dof)###
    nAllocation=zeros(m,1)
    for j=1:m
        sum=0
        for i=1:(n-1)
            sum=x[i,j]+sum
        end
        nAllocation[j]=e[j]-sum
        utility[n]=α[j]*(nAllocation[j]^(ω[n,j]-1))/(ω[n,j]-1)+utility[n]
    end
    utility[n]=λ[n]*utility[n]
    socialUtility=sum(utility)
    return socialUtility
end

h=10^(-6)
function CenteredDifference(objective,x,h)
    ##Note that the point x is a matrix in general. The coordinates of the gradient should 
    ##correspond to the coordinates of the vectorization of the matrix
    N=size(x)[1]
    M=size(x)[2]

    gradient=ones(N,M)
    step=zeros(N,M)
    step=zeros(N,M)
    forwardPoint=zeros(N,M)
    backwardPoint=zeros(N,M)
    for i=1:N
        for j=1:M
            mStep=zeros(N,M)
            mStep[i,j]=h
            forwardPoint=x+mStep
            backwardPoint=x-mStep
            gradient[i,j]=(objective(forwardPoint,α,ω,λ,e)-objective(backwardPoint,α,ω,λ,e))/(2*h)
        end
    end
    return gradient
end


function GridSearch(objective,grid)
    N=size(grid)[3]
    maximizer=grid[:,:,1]

        for i=1:N
               if objective(grid[:,:,i],α,ω,λ,e)>objective(maximizer,α,ω,λ,e)
                    maximizer=grid[:,:,i]
                end
        end

    return maximizer
end

tolerance=10^(-7)
θ=0.2
###Set θ=0.1 for the n=m=10 case###

xInitial=zeros(n-1,m)
for j=1:m
    for i=1:size(xInitial)[1]
        xInitial[i,j]=e[j]/n
    end
end

###############################
###Main optimization routine###
###############################


function SteepestDescent(objective,xInitial,θ,tolerance)
    tic=@elapsed begin

    N=size(xInitial)[1]
    M=size(xInitial)[2]
    xGuess=xInitial

    D=1
    while D>tolerance
        U=objective(xGuess,α,ω,λ,e)
        gradient=CenteredDifference(objective,xGuess,h)

        maxStep=1/(100*maximum([abs(minimum(gradient)),maximum(gradient)]))
        gridSize=10000
        grid=zeros(N,M,gridSize)
        δ=abs(maxStep)/gridSize


        for i=1:gridSize
            grid[:,:,i]=xGuess+gradient*δ*i
        end

        xNew=GridSearch(objective,grid)
        xNew=map((x)->maximum([x,0]),xNew)
        D=abs(objective(xGuess,α,ω,λ,e)-objective(xNew,α,ω,λ,e))
        println(D)
        xGuess=θ*xNew+(1-θ)*xGuess
    end
    end

    mAllocation=ones(N+1,M)
    mAllocation[1:N,:]=xGuess
    for j=1:M
        mAllocation[N+1,j]=e[j]-sum(xGuess[:,j])
    end
    println(tic)
    return mAllocation

end


