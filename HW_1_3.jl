############################
###Numerical optimization###
############################

###Script minimizes objective function using Newton, BFGS, steepest
###descent, and conjugate descent algorithms, and computes run time
###for each### 

function main()

tol=10^(-8)


function Objective(x)

    return 100*(x[2]-x[1]^2)^2+(1-x[1])^2

end

function Gradient(x)

    return [-400*x[2]*x[1]+400*x[1]^3+2*x[1]-2, 200*x[2]-200*x[1]^2]

end

function Hessian(x)

    Hf=zeros(2,2)
    Hf[1,1]=-400*x[2]+1200*x[1]^2+2
    Hf[1,2]=-400*x[1]
    Hf[2,1]=-400*x[1]
    Hf[2,2]=200
    
    return Hf

end

xInitial=[0,0]
function Newton(Objective,xInitial,tol,θ)
    tic=@elapsed begin

    ###Initialization##
    xGuess=xInitial
    Df=Gradient(xInitial)
    Hf=Hessian(xInitial)
    D=1

    ###Recursion###
    while D>tol
        xNew=xGuess-(Hf)^(-1)*Df
        D=abs(Objective(xNew)-Objective(xGuess))
        xGuess=θ*xNew+(1-θ)*xGuess

        Df=Gradient(xGuess)
        Hf=Hessian(xGuess)
    end
    end

    return hcat(xGuess',norm(xGuess-ones(2,1)),tic)
    
end


function GridSearch(Objective,grid)

    ###Use for minimization###
    M=size(grid)[2]
    optimum=grid[:,1]

    for i=1:M
        if Objective(optimum)>Objective(grid[:,i])
            optimum=grid[:,i]
        end
    end
    
    return optimum
    
end

θ=0.3

function BFGS(Objective,xInitial,tol,θ)
    tic=@elapsed begin

    ###Initialization###
    xGuess=xInitial
    N=size(xInitial)[1]
    Df=Gradient(xGuess)
    Q=Matrix(I,N,N) 
    D=1
    
    ###Recursion###
    while D>tol
        ###Create grid for line search###
        gridSize=100000
        Δ=Q*Df/(gridSize)
        grid=zeros(N,gridSize)   ##Line extends in both directions
        grid[:,1]=xGuess-Δ*gridSize/2

        for i=2:gridSize
            grid[:,i]=grid[:,i-1]+Δ
        end

        ###Update x with grid search###
        xNew=θ*GridSearch(Objective,grid)+(1-θ)*xGuess
        DfNew=Gradient(xNew)
        
        γ=DfNew-Df
        δ=xNew-xGuess
        Q=Q-((δ*transpose(γ)*Q+Q*γ*transpose(δ))/(transpose(δ)*γ))+
        (1+(transpose(γ)*Q*γ)/(transpose(δ)*γ))*((δ*transpose(δ))/(transpose(δ)*γ))

        
        D=abs(Objective(xNew)-Objective(xGuess))
        xGuess=xNew
        Df=DfNew
    end
    end
    
    return hcat(xGuess',norm(xGuess-ones(2,1)),tic)
end


function SteepestDescent(Objective,xInitial,tol,θ)
    tic=@elapsed begin

    ###Initialization###
    xGuess=xInitial
    N=size(xInitial)[1]
    Df=Gradient(xGuess)
    D=1

    while D>tol
        ###Create grid for line search###
        maxStep=1/(10*maximum([abs(minimum(Df)),maximum(Df)]))
        gridSize=10000
        grid=zeros(N,gridSize)
        Δ=Df/gridSize

        for i=1:gridSize
            grid[:,i]=xGuess-Δ*i
        end

        ###Update with line search###
        xNew=θ*GridSearch(Objective,grid)+(1-θ)*xGuess
        DfNew=Gradient(xNew)
        D=abs(Objective(xNew)-Objective(xGuess))
        xGuess=xNew
        Df=DfNew
    end
    end

    return hcat(xGuess',norm(xGuess-ones(2,1)),tic)

end


function ConjugateDescent(Objective,xInitial,tol)
    tic=@elapsed begin

    ###Initialization###
    xGuess=xInitial
    N=size(xInitial)[1]
    Df=Gradient(xGuess)
    d=Df
    D=1

    while D>tol
        ###Create grid for line search###
        gridSize=10000
        grid=zeros(N,gridSize)
        Δ=d/gridSize
        grid[:,1]=xGuess-d/2

        for i=2:gridSize
            grid[:,i]=grid[:,i-1]+Δ
        end

        ###Update with line search###
        xNew=GridSearch(Objective,grid)
        D=maximum([abs(Objective(xNew)-Objective(xGuess)),norm(xNew-xGuess)])
        
        DfNew=Gradient(xNew)
        β=(DfNew'*(DfNew-Df))/(Df'*Df)
        β=maximum([0,β])
        d=-DfNew+β*d

        xGuess=xNew
        Df=DfNew
    end
    end

    return hcat(xGuess',norm(xGuess-ones(2,1)),tic)

end

###1)Newton 2)BFGS 3)Steepest Descent 4)Conjugate Descent
tabular(vcat(Newton(Objective,xInitial,tol,θ),BFGS(Objective,xInitial,tol,θ),
SteepestDescent(Objective,xInitial,tol,θ),ConjugateDescent(Objective,xInitial,tol)))

end
