####################################################
### #Projection with Chebyshev Polynomials for RBC #
### #Alexander Sawyer                              #
### #Fall 2021                                     #
####################################################

using NLsolve
using Roots
using Plots
using Interpolations

import Base.+  
+(f::Function, g::Function) = (x) -> f(x) + g(x)

###Initialize TFP Markov process, steady state, 
###and state space###

α=1/3
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


function ChebyApprox(order,coefficientVector)
    
    T0(x)=1
    T1(x)=x
    
    chebyVector=Array{Function,1}(undef,order)
  
    chebyVector[1]=T0
    chebyVector[2]=T1

    for i=3:order

        chebyVector[i]=x->2*x*chebyVector[i-1](x)-chebyVector[i-2](x)

    end

    polynomial=x->sum([chebyVector[i](x)*coefficientVector[i] for i=1:order])

    return polynomial

end


function ChebyDerivative(order,coefficientVector)

    T0prime(x)=0.0
    T1prime(x)=1.0
    T0(x)=1.0
    T1(x)=x

    derivativeVector=Array{Function,1}(undef,order)
    chebyVector=Array{Function,1}(undef,order)

    derivativeVector[1]=T0prime
    derivativeVector[2]=T1prime
    chebyVector[1]=T0
    chebyVector[2]=T1

    for i=3:order

        chebyVector[i]=x->2*x*chebyVector[i-1](x)-chebyVector[i-2](x)

        derivativeVector[i]=x->2*x*derivativeVector[i-1](x)+2*chebyVector[i-1](x)-
        derivativeVector[i-2](x)

    end

    polynomial=x->sum([derivativeVector[i](x)*coefficientVector[i] for i=1:order])

    return polynomial


end


function ChangeOfVariable(a,b,x)

    y=2*((x-a)/(b-a))-1

end

###Feed in the vectorized coefficient matrix for the entire system

function ResidualCheby(vCoefficient,k,zIndex,order)

    k_hat=ChangeOfVariable(gridCapitalMin,gridCapitalMax,k)

    ###First transformed vectorized coefficient matrix back into a
    ### matrix for readability: [θ^V1,,,θ^V3,θ^c1,...,θ^c3]

    mCoefficient=hcat(vCoefficient[1:order],vCoefficient[order+1:2*order],
    vCoefficient[2*order+1:3*order],vCoefficient[3*order+1:4*order],
    vCoefficient[4*order+1:5*order],vCoefficient[5*order+1:6*order])

    residualValue=zeros(2)

    valueFunction=Array{Function,1}(undef,3)
    #consumptionFunction=Array{Function,1}(undef,3)
    valueDerivative=Array{Function,1}(undef,3)
    #vLabor=Array{Function,1}(undef,3)

    for i=1:3
        
        valueFunction[i]=x->ChebyApprox(order,mCoefficient[:,i])(x)
        valueDerivative[i]=x->ChebyDerivative(order,mCoefficient[:,i])(x)
        #consumptionFunction[i]=x->ChebyApprox(order,mCoefficient[:,i+3])(x)
        #vLabor[i]=x->(maximum([exp(gridTfp[i])*(1-α)*
        #x^α/consumptionFunction[i](ChangeOfVariable(gridCapitalMin,gridCapitalMax,x)),0]))^(1/(1+α))

    end

    consumption=x->ChebyApprox(order,mCoefficient[:,3+zIndex])(x)
    #labor=maximum([(exp(gridTfp[zIndex])*(1-α)*(k^α/consumption(k_hat))),0])^(1/(1+α))
    labor=(exp(gridTfp[zIndex])*(1-α)*k^α/consumption(k_hat))^(1/(1+α))
    kNext=(1-α)*exp(gridTfp[zIndex])*(k/labor)^α*labor+α*exp(gridTfp[zIndex])*(labor/k)^(1-α)*k-
    consumption(k_hat)+(1-δ)*k
    #kNext=exp(gridTfp[zIndex])*k^α*labor^(1-α)-consumption(k_hat)+(1-δ)*k
    k_hatNext=ChangeOfVariable(gridCapitalMin,gridCapitalMax,kNext)


    ###The residual functions are Bellman [1] and Euler [2]

    residualValue[1]=valueFunction[zIndex](k_hat)-log(consumption(k_hat))+(labor^2/2)-
    β*sum([ΠTfp[zIndex,i]*valueFunction[i](k_hatNext) for i=1:3])

    residualValue[2]=(1/consumption(k_hat))-β*(2/(gridCapitalMax-gridCapitalMin))*
    sum([ΠTfp[zIndex,i]*valueDerivative[i](k_hatNext) for i=1:3])
    #residualValue[2]=(1/consumption(k_hat))-β*sum([ΠTfp[zIndex,i]*α*exp(gridTfp[i])*
    #(vLabor[i](k_hatNext)/kNext)^(1-α)*(1/consumptionFunction[i](k_hatNext)) for i=1:3])

    return residualValue

end


function CoefficientSolveCheby(order,evaluationPoints,firstGuess)

    function f!(F,vCoefficient)
        
        for kIndex=1:order

            k_hat=evaluationPoints[kIndex]

            k=((k_hat+1)/2)*(gridCapitalMax-gridCapitalMin)+gridCapitalMin

            for zIndex=1:3

                FIndex1=(kIndex-1)*6+zIndex
                FIndex2=(kIndex-1)*6+zIndex+3
                F[FIndex1]=ResidualCheby(vCoefficient,k,zIndex,order)[1]
                F[FIndex2]=ResidualCheby(vCoefficient,k,zIndex,order)[2]

            end

        end

    end


    return nlsolve(f!,firstGuess).zero

end



function MultistepCheby(order)

    firstGuess=zeros(3)
    firstGuess[1]=1
    firstGuessStack=firstGuess
    for i=2:6

        firstGuessStack=vcat(firstGuessStack,firstGuess)

    end

    firstEvaluationPoints=[cos((1/6)*π); cos((3/6)*π); cos((5/6)*π)]
    #firstEvaluationPoints=[-1,0,1]
    vfirstCoefficient=CoefficientSolveCheby(3,firstEvaluationPoints,firstGuessStack)

    mfirstCoefficients=hcat(vfirstCoefficient[1:3],vfirstCoefficient[3+1:2*3],
    vfirstCoefficient[2*3+1:3*3],vfirstCoefficient[3*3+1:4*3],
    vfirstCoefficient[4*3+1:5*3],vfirstCoefficient[5*3+1:6*3])

    msecondGuess=vcat(mfirstCoefficients,zeros(order-3,6))
    vsecondGuess=vec(msecondGuess)

    evaluationPoints=zeros(order)

    for i=1:order

       evaluationPoints[i]=cos(((2*i-1)/(2*order))*π) 
       #evaluationPoints[i]=cos(((i-1)/(order-1))*π)

    end


    vfinalCoefficient=CoefficientSolveCheby(order,evaluationPoints,vsecondGuess)

    mfinalCoefficient=hcat(vfinalCoefficient[1:order],vfinalCoefficient[order+1:2*order],
    vfinalCoefficient[2*order+1:3*order],vfinalCoefficient[3*order+1:4*order],
    vfinalCoefficient[4*order+1:5*order],vfinalCoefficient[5*order+1:6*order])

    return mfinalCoefficient

end

function ProjectedCheby(order)

    mCoefficients=MultistepCheby(order)

    Consumption=Array{Function,1}(undef,3)
    Value=Array{Function,1}(undef,3)
    Labor=Array{Function,1}(undef,3)
    CapitalSaving=Array{Function,1}(undef,3)
    Output=Array{Function,1}(undef,3)


    for zIndex=1:3

        Consumption[zIndex]=x->ChebyApprox(order,mCoefficients[:,zIndex+3])(ChangeOfVariable(gridCapitalMin,
        gridCapitalMax,x))
        Value[zIndex]=x->ChebyApprox(order,mCoefficients[:,zIndex])(ChangeOfVariable(gridCapitalMin,
        gridCapitalMax,x))
        Labor[zIndex]=x->(exp(gridTfp[zIndex])*(1-α)*x^α/Consumption[zIndex](x))^(1/(1+α))
        CapitalSaving[zIndex]=x->(1-α)*exp(gridTfp[zIndex])*(x/Labor[zIndex](x))^α*Labor[zIndex](x)+
        α*exp(gridTfp[zIndex])*(Labor[zIndex](x)/x)^(1-α)*x-Consumption[zIndex](x)+(1-δ)*x
        Output[zIndex]=x->exp(gridTfp[zIndex])*x^α*Labor[zIndex](x)^(1-α)

    end

    fValue=(x::Float64,z::Int64)->Value[z](x)
    fConsumption=(x::Float64,z::Int64)->Consumption[z](x)
    fCapital=(x::Float64,z::Int64)->CapitalSaving[z](x)
    fLabor=(x::Float64,z::Int64)->Labor[z](x)
    fOutput=(x::Float64,z::Int64)->Output[z](x)
    fWage=(x::Float64,z::Int64)->exp(gridTfp[z])*α*(fCapital(x,z)/fLabor(x,z))^α
    fReturn=(x::Float64,z::Int64)->exp(gridTfp[z])*α*(fLabor(x,z)/fCapital(x,z))^(1-α)

    return fCapital,fConsumption,fOutput,fLabor,fWage,fReturn,fValue,Consumption


end