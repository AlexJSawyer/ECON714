####################################################
### #Projection with Finite Elements for RBC       #
### #Alexander Sawyer                              #
### #Fall 2021                                     #
####################################################

using NLsolve
using Roots
using Plots
using Interpolations

α=0.3333
β=0.97
γ=1
δ=0.1

function SteadyState()

    ### x[1] is tomorrow's capital x[2] is labor ###

    ### F[1] is the Euler equation and F[2] labor as a 
    ### function of tomorrow's capital ###

    function f(F,x)

        F[1]=β*(α*(x[2]/x[1])^(1-α)+(1-δ))-1
        F[2]=(1-α)*(x[1]/x[2])^(α)*x[2]^2+(α*(x[2]/x[1])^(1-α)*x[1]-
        (x[1]-(1-δ)*x[1]))*x[2]-(1-α)*(x[1]/x[2])^(α)

    end

    return nlsolve(f,[3.0;1.0]).zero    


end



steadyState=SteadyState()
capitalSS=steadyState[1]
laborSS=steadyState[2]
consumptionSS=(1-α)*(capitalSS/laborSS)^α*laborSS+α*(laborSS/capitalSS)^(1-α)*capitalSS-0.1*capitalSS

gridCapitalMin=0.7*capitalSS
gridCapitalMax=1.3*capitalSS

gridTfp=[-0.05,0,0.05]
ΠTfp=[0.97 0.03 0;0.01 0.98 0.01;0 0.03 0.97]


### Define generic piecewise basis functions ###

function GenericBasis(x,i,order,elementGrid)

    output=0

    if x≥elementGrid[i-1]&&x≤elementGrid[i]

        output=(x-elementGrid[i-1])/(elementGrid[i]-elementGrid[i-1])

    end

    if x≥elementGrid[i]&&x≤elementGrid[i+1]

        output=(elementGrid[i+1]-x)/(elementGrid[i+1]-elementGrid[i])

    end

    return output

end


function FirstBasis(x,order,elementGrid)

    output=0

    if x≥elementGrid[1]&&x≤elementGrid[2]

        output=(elementGrid[2]-x)/(elementGrid[2]-elementGrid[1])

    end

    return output

end

function LastBasis(x,order,elementGrid)

    output=0

    if x≥elementGrid[order-1]&&x≤elementGrid[order]

        output=(x-elementGrid[order-1])/(elementGrid[order]-elementGrid[order-1])
    
    end

    return output

end


function ElementBuild(order,gridCapitalMin,gridCapitalMax)

    gridSize=order
    grid=zeros(gridSize)

    for i=1:gridSize

        grid[i]=gridCapitalMin+((gridCapitalMax-gridCapitalMin)/(gridSize-1))*(i-1)

    end

    return grid

end


function FEApprox(elementGrid,vCoefficients)

    order=length(elementGrid)
    vBasis=Array{Function,1}(undef,order)


    ### populate basis vector ###

    vBasis[1]=x->FirstBasis(x,order,elementGrid)
    vBasis[order]=x->LastBasis(x,order,elementGrid)

    for i=2:order-1

        vBasis[i]=x->GenericBasis(x,i,order,elementGrid)

    end

    fOutput=x->sum([vCoefficients[i]*vBasis[i](x) for i=1:order])

    return fOutput

end


###Input coefficient vector as a matrix, each state a column ###

function Residual(k,zIndex,elementGrid,mCoefficients)

    order=length(elementGrid)

    ### 1) build consumption function with FE and resulting labor ###
    ###    and saving functions ###                                  

    vConsumption=Array{Function,1}(undef,3)
    vLabor=Array{Function,1}(undef,3)
    vCapital=Array{Function,1}(undef,3)
    
    for i=1:3
        
        vConsumption[i]=x->FEApprox(elementGrid,mCoefficients[:,i])(x)
        vLabor[i]=x->(exp(gridTfp[i])*(1-α)*(x^α/maximum([(vConsumption[i](x)),0.0001])))^(1/(1+α))
        vCapital[i]=x->(1-α)*exp(gridTfp[i])*(x/vLabor[i](x))^α*vLabor[i](x)+
                    α*exp(gridTfp[i])*(vLabor[i](x)/x)^(1-α)*x-vConsumption[i](x)+(1-δ)*x

    end


    ### 2) Construct residual values ###

    c=maximum([vConsumption[zIndex](k),0.001])
    l=vLabor[zIndex](k)
    kNext=maximum([vCapital[zIndex](k),0.00001])

    kernel=zeros(3)

    for i=1:3

        lNext=vLabor[i](kNext)
        cNext=maximum([vConsumption[i](kNext),0.00005])

        kernel[i]=(1+α*exp(gridTfp[i])*(lNext/kNext)^(1-α)-δ)*(1/cNext)

    end

    E1=kernel'*ΠTfp[zIndex,:]

    residual=1/c-β*E1[1]

    return residual





end


function Galerkin(elementGrid,mCoefficients)

    ### 1) build consumption function with FE and resulting labor ###
    ###    and saving functions ###                                  

    order=length(elementGrid)
    
    vConsumption=Array{Function,1}(undef,3)
    vLabor=Array{Function,1}(undef,3)
    vCapital=Array{Function,1}(undef,3)
    
    for i=1:3
        
        vConsumption[i]=x->FEApprox(elementGrid,mCoefficients[:,i])(x)
        vLabor[i]=x->(exp(gridTfp[i])*(1-α)*(x^α/maximum([(vConsumption[i](x)),0.001])))^(1/(1+α))
        vCapital[i]=x->(1-α)*exp(gridTfp[i])*(x/maximum([vLabor[i](x),0.00001]))^α*vLabor[i](x)+
                    α*exp(gridTfp[i])*(vLabor[i](x)/x)^(1-α)*x-vConsumption[i](x)+0.9*x

    end

    
    ### 2) Build the weight functions and the integrand ###

    vWeights=Array{Function,1}(undef,order)


    for i=2:order-1

        vWeights[i]=x->GenericBasis(x,i,order,elementGrid)

    end

    vWeights[1]=x->FirstBasis(x,order,elementGrid)
    vWeights[order]=x->LastBasis(x,order,elementGrid)

    mIntegrand=Array{Function,2}(undef,order,3)

    for j=1:3

        for i=1:order

            mIntegrand[i,j]=x->vWeights[i](x)*Residual(x,j,elementGrid,mCoefficients)

        end

    end

    
    ### Quadrature for integrals ###

    function trapezoid(integrand,a,b,N)
    
        h=(b-a)/N
        integral=integrand(a)/2
        for i in 1:(N-1)
            x=h*(i-1)+a
            integral=integral+integrand(x)
        end
        integral=integral+integrand(b)/2
        integral=integral*h

    
        return integral
    
    end


    function simpson(integrand,domainMin,domainMax,N)
        
        h=(domainMax-domainMin)/N
    
        ###Add the irregular summands###
        integral=integrand(domainMin)
        integral=integral+integrand(domainMax)
    
        ###Add the regular summands###
        for i=1:(N/2)-1
            t0=(2*i+1)*h+domainMin
            t1=(2*i)*h+domainMin
            integral=integral+2*integrand(t0)+4*integrand(t1)
        end
    
        integral=integral*h/3
        
    
        return integral
    
    end


    ### 3) compute the matrix of definite integrals ###

    mIntegrals=zeros(order,3)

    for j=1:3

        mIntegrals[1,j]=trapezoid(mIntegrand[1,j],elementGrid[1],elementGrid[2],500)

        for i=2:order-1

            mIntegrals[i,j]=trapezoid(mIntegrand[i,j],elementGrid[i-1],elementGrid[i+1],500)

        end

        mIntegrals[order,j]=trapezoid(mIntegrand[order,j],elementGrid[order-1],elementGrid[order],500)

    end

    

    return mIntegrals

end


### Coefficient Solver ###

function CoefficientSolve(elementGrid,firstGuess)

    order=length(elementGrid)

    function f(F,vCoefficients)

        mCoefficients=zeros(order,3)

        for j=1:3

            mCoefficients[:,j]=vCoefficients[(j-1)*order+1:j*order]

        end

        mGalerkin=Galerkin(elementGrid,mCoefficients)

        for j=1:3

            for i=1:order

                FIndex=(j-1)*order+i
                F[FIndex]=mGalerkin[i,j]

            end

        end

    end

    tic=@elapsed begin
    output=nlsolve(f,firstGuess).zero
    end
    println(tic)
    println(output)
    return output

end



function Multistep(finalElementCount)

    tic=@elapsed begin

    if mod(finalElementCount,2)==0
        
        firstElementCount=convert(Int64,finalElementCount/2)

    else 

        println("Final element count must be even")

    end

    firstOrder=firstElementCount+1
    firstElementGrid=ElementBuild(firstOrder,gridCapitalMin,gridCapitalMax)
    firstGuess=[1+(i-1)/(firstOrder-1) for i=1:firstOrder]
    vFirstGuessStack=vcat(firstGuess,firstGuess,firstGuess)

    vFirstCoefficient=CoefficientSolve(firstElementGrid,vFirstGuessStack)

    mFirstCoefficient=hcat(vFirstCoefficient[1:firstOrder],vFirstCoefficient[firstOrder+1:2*firstOrder],
    vFirstCoefficient[2*firstOrder+1:3*firstOrder])

    
    finalOrder=finalElementCount+1
    finalElementGrid=ElementBuild(finalOrder,gridCapitalMin,gridCapitalMax)
    mSecondGuess=zeros(finalOrder,3)

    for j=1:3

        mSecondGuess[1,j]=mFirstCoefficient[1,j]
        oldIndex=0

        for i=2:finalOrder

            if mod(i,2)==0

                oldIndex=oldIndex+1

            end

            mSecondGuess[i,j]=(mFirstCoefficient[oldIndex+1,j]-mFirstCoefficient[oldIndex,j])
            +mSecondGuess[i-1,j]

        end

    end

    vSecondGuess=vcat(mSecondGuess[:,1],mSecondGuess[:,2],mSecondGuess[:,3])

    vFinalCoefficient=CoefficientSolve(finalElementGrid,vSecondGuess)

    mFinalCoefficients=hcat(vFinalCoefficient[1:finalOrder],vFinalCoefficient[finalOrder+1:2*finalOrder]
    ,vFinalCoefficient[2*finalOrder+1:3*finalOrder])

    
    end
    println(tic)


    return mFinalCoefficients


end



function FunctionBuild(elementNumber)

    mCoefficients=Multistep(elementNumber)

    #firstElementGrid=ElementBuild(elementNumber,gridCapitalMin,gridCapitalMax)
    #firstGuess=[1+0.5*(i-1) for i=1:elementNumber]
    #vFirstGuessStack=vcat(firstGuess,firstGuess,firstGuess)

    #vFirstCoefficient=CoefficientSolve(firstElementGrid,vFirstGuessStack)

    #mCoefficients=hcat(vFirstCoefficient[1:elementNumber],
    #vFirstCoefficient[elementNumber+1:2*elementNumber]
    #,vFirstCoefficient[2*elementNumber+1:3*elementNumber])

    vConsumption=Array{Function,1}(undef,3)
    vLabor=Array{Function,1}(undef,3)
    vCapital=Array{Function,1}(undef,3)
    vOutput=Array{Function,1}(undef,3)
    elementGrid=ElementBuild(elementNumber+1,gridCapitalMin,gridCapitalMax)
    
    for i=1:3
        
        vConsumption[i]=x->FEApprox(elementGrid,mCoefficients[:,i])(x)
        vLabor[i]=x->(exp(gridTfp[i])*(1-α)*(x^α/maximum([(vConsumption[i](x)),0.000001])))^(1/(1+α))
        vCapital[i]=x->(1-α)*exp(gridTfp[i])*(x/maximum([vLabor[i](x),0.000001]))^α*vLabor[i](x)+
                    α*exp(gridTfp[i])*(vLabor[i](x)/x)^(1-α)*x-vConsumption[i](x)+0.9*x
        vOutput[i]=x->exp(gridTfp[i])*vLabor[i](x)^(1-α)*x^α

    end

    fConsumption=(x::Float64,z::Int64)->vConsumption[z](x)
    fCapital=(x::Float64,z::Int64)->vCapital[z](x)
    fLabor=(x::Float64,z::Int64)->vLabor[z](x)
    fOutput=(x::Float64,z::Int64)->vOutput[z](x)
    fWage=(x::Float64,z::Int64)->exp(gridTfp[z])*0.67*(fCapital(x,z)/fLabor(x,z))^0.33
    fReturn=(x::Float64,z::Int64)->exp(gridTfp[z])*0.33*(fLabor(x,z)/fCapital(x,z))^0.67

    return fCapital,fConsumption,fOutput,fLabor,fWage,fReturn


end
















