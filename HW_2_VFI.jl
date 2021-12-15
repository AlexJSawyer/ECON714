####################################################################
## #Value function Iteration for RBC with Mulitgrid & accelerator #
## #Alexander Sawyer                                              #
## #Fall 2021                                                     #
###################################################################

using NLsolve
using Roots
using Plots
using Interpolations



##########################################
###Value Function main routine############
# Input max operator rate and grid size###
# Output value function and policy grids##
##########################################

maxOperatorRate=1/10
gridSize=1000
valueFunctionGuess=zeros(gridSize,3)

for zIndex=1:3

    for kIndex=1:gridSize

        valueFunctionGuess[kIndex,zIndex]=-10+sqrt(kIndex)

    end

end

gridTfp=[-0.05,0,0.05]
ΠTpf=[0.97 0.03 0;0.01 0.98 0.01;0 0.03 0.97]

function SteadyState()

    ### x[1] is tomorrow's capital x[2] is labor ###

    ### F[1] is the Euler equation and F[2] labor as a 
    ### function of tomorrow's capital ###

    function f(F,x)

        F[1]=0.97*((1/3)*(x[2]/x[1])^(2/3)+0.9)-1
        F[2]=(2/3)*(x[1]/x[2])^(1/3)*x[2]^2+((1/3)*(x[2]/x[1])^(2/3)*x[1]-
        (x[1]-0.9*x[1]))*x[2]-(2/3)*(x[1]/x[2])^(1/3)

    end

    return nlsolve(f,[3.0;1.0]).zero    


end

steadyState=SteadyState()
capitalSS=steadyState[1]
laborSS=steadyState[2]

gridCapitalMin=0.7*capitalSS
gridCapitalMax=1.3*capitalSS



function VFI(maxOperatorRate,gridSize,valueFunctionGuess)

    ### Build grids--first need labor rule ####

    
    gridCapital=zeros(gridSize)

    capitalPolicy=zeros(gridSize,3)
    capitalPolicyIndex=zeros(gridSize,3)
    valueFunction=zeros(gridSize,3)

    for i=1:gridSize

        gridCapital[i]=gridCapitalMin+((gridCapitalMax-gridCapitalMin)/(gridSize-1))*(i-1)

    end

    for zIndex=1:3

        for kIndex=1:gridSize

            valueFunction[kIndex,zIndex]=valueFunctionGuess[kIndex,zIndex]

        end

    end


    ###Specifiy labor implicitly as a funciton of k'###

    function LaborPolicy(k,kNext,z)

        g(x)=exp(z)*(2/3)*(k/x)^(1/3)*x^2+((exp(z)/3)*(x/k)^(2/3)*k-(kNext-0.9*k))*x-
        exp(z)*(2/3)*(k/x)^(1/3)

        return find_zero(g,(0.0001,8))

    end


        #function g(F,x)

         #   F[1]=exp(z)*(0.66)*(k/x[1])^(0.33)*x[1]^2+((exp(z)/3)*(x[1]/k)^(0.33)*k-(kNext-0.9*k))*x[1]-
          #  exp(z)*(0.66)*(k/x[1])^(0.33)

       # end

        #return nlsolve(g,[lss]).zero[1]

    #end

    function BellmanRHS(k,kNextIndex,zIndexx,valueFunction)

        kNext=gridCapital[kNextIndex]
        z=gridTfp[zIndexx]
        l=LaborPolicy(k,kNext,z)

        ###The nonlinear solver for labor might yield a complex solution,
        ### in which case assign a low value and exit the program   
            w=exp(z)*(2/3)*(k/l)^(1/3)
            r=exp(z)*(1/3)*(l/k)^(2/3)

            expectedValueFunction=ΠTpf[zIndexx,:]'*valueFunction[kNextIndex,:]
            logArgument=maximum([w*l+r*k-(kNext-0.9*k),0.00001])
            RHS=log(logArgument)-(l^2)/2 .+0.97*expectedValueFunction

        return RHS

    end


    ########################################
    ###The VFI iteration loop begins here###
    ########################################
    
    vfiIndex=0
    policyruncount=0
    tolerance=10^(-7)
    DD=1
    
    while DD>tolerance &&DD<100

        ###Policy function update###
        if mod(vfiIndex,(1/maxOperatorRate))==0.0

            for zIndex=1:3

                for kIndex=1:gridSize

                    valueSearch=zeros(gridSize)
                    winningIndex=1
                    winningValue=BellmanRHS(gridCapital[kIndex],1,zIndex,valueFunction)

                    for i=2:gridSize

                        if BellmanRHS(gridCapital[kIndex],i,zIndex,valueFunction)>winningValue
                            winningIndex=i
                            winningValue=BellmanRHS(gridCapital[kIndex],i,zIndex,valueFunction)
                        end

                        #valueSearch[i]=BellmanRHS(gridCapital[kIndex],i,zIndex,valueFunction)
                    end

                    capitalPolicyIndex[kIndex,zIndex]=winningIndex
                    #capitalPolicyIndex[kIndex,zIndex]=convert(Int64,findmax(valueSearch)[2])
                    capitalPolicy[kIndex,zIndex]=gridCapital[convert(Int64,capitalPolicyIndex[kIndex,zIndex])]

                end

                
            
            end

            policyruncount=policyruncount+1
        end



        ###Value function update###

        valueFunctionNew=zeros(gridSize,3)

        for zIndex=1:3

            for kIndex=1:gridSize

                valueFunctionNew[kIndex,zIndex]=BellmanRHS(gridCapital[kIndex],
                convert(Int64,capitalPolicyIndex[kIndex,zIndex]),zIndex,valueFunction)
                
            end

        end


        ###Check convergence and update gueses###

        DD=maximum(abs.(valueFunctionNew-valueFunction))
        
        for kIndex=1:gridSize

            for zIndex=1:3
                
                valueFunction[kIndex,zIndex]=valueFunctionNew[kIndex,zIndex]

            end
            
        end
        
        vfiIndex=vfiIndex+1

        println(DD)

    end

 
    return valueFunction, capitalPolicy, capitalPolicyIndex, policyruncount




end
    
function Multigrid(firstGridOrder,lastGridOrder,maxOperatorRate,scale)

    tic=@elapsed begin

    finalCapitalGridSize=convert(Int64,scale*10^(lastGridOrder))
    finalCapitalGrid=zeros(finalCapitalGridSize)
    
    valueFunctionGuess=zeros(convert(Int64,scale*10^firstGridOrder),3)
    gridSize=convert(Int64,scale*10^firstGridOrder)

    for zIndex=1:3

        for kIndex=1:gridSize

            valueFunctionGuess[kIndex,zIndex]=-10+sqrt(kIndex)

        end

    end

    NumberOfRuns=lastGridOrder-(firstGridOrder-1)

    VFISolutions=Array{Array,2}(undef,3,2)

    for runNumber=1:NumberOfRuns

        runOrder=(runNumber-1)+firstGridOrder
        
        valueFunction=zeros(convert(Int64,scale*10^runOrder),3)
        capitalPolicy=zeros(convert(Int64,scale*10^runOrder),3)

        VFISolution=VFI(maxOperatorRate,convert(Int64,scale*10^runOrder),valueFunctionGuess)
        valueFunction=VFISolution[1]
        capitalPolicy=VFISolution[2]

        
        ###use current iteration solution to interpolate a guess for next iteration###

        newCapitalGridSize=convert(Int64,scale*10^(runOrder+1))
        oldCapitalGridSize=convert(Int64,scale*10^(runOrder))
        newCapitalGrid=zeros(newCapitalGridSize)
        oldCapitalGrid=zeros(oldCapitalGridSize)

        

        gridCapitalMin=0.7*capitalSS
        gridCapitalMax=1.3capitalSS

        for i=1:oldCapitalGridSize

            oldCapitalGrid[i]=gridCapitalMin+
            ((gridCapitalMax-gridCapitalMin)/(oldCapitalGridSize-1))*(i-1)
    
        end

        for i=1:newCapitalGridSize

            newCapitalGrid[i]=gridCapitalMin+
            ((gridCapitalMax-gridCapitalMin)/(newCapitalGridSize-1))*(i-1)
    
        end

        if runNumber==NumberOfRuns
            finalCapitalGrid=oldCapitalGrid
        end


        valueFunctionGuess=zeros(newCapitalGridSize,3)

        for zIndex=1:3
            
            valueInterp=LinearInterpolation(oldCapitalGrid,valueFunction[:,zIndex])
            valueFunctionGuess[:,zIndex]=valueInterp(newCapitalGrid)

        end

        VFISolutions[runNumber,1]=valueFunction
        VFISolutions[runNumber,2]=capitalPolicy

        println(runNumber)
        
    end

    valueFunctionFinal=VFISolutions[NumberOfRuns,1]
    capitalPolicyFinal=VFISolutions[NumberOfRuns,2]

    end
    println(tic)



    function fValue(x,s) 

        interpLin=LinearInterpolation(finalCapitalGrid,valueFunctionFinal[:,s],extrapolation_bc=Line())
        return interpLin(x)

    end

    function fCapital(x::Float64,s::Int64)

        interpLin=LinearInterpolation(finalCapitalGrid,capitalPolicyFinal[:,s],extrapolation_bc=Line())
        return interpLin(x)

    end

    function fLabor(k::Float64,z::Int64)

        kNext=fCapital(k,z)

        g(x)=exp(gridTfp[z])*(2/3)*(k/x)^(1/3)*x^2+((exp(gridTfp[z])/3)*(x/k)^(2/3)*k-(kNext-0.9*k))*x-
        exp(gridTfp[z])*(2/3)*(k/x)^(1/3)

        return find_zero(g,(0.0001,8))

    end

    function fConsumption(k::Float64,z::Int64)

        r=exp(gridTfp[z])*0.33*(fLabor(k,z)/fCapital(k,z))^(0.67)
        w=exp(gridTfp[z])*0.67*(fCapital(k,z)/fLabor(k,z))^0.33

        output=w*fLabor(k,z)+(0.9+r)*k-fCapital(k,z)
        return output

    end

    fWage=(k::Float64,z::Int64)->exp(gridTfp[z])*0.67*(fCapital(k,z)/fLabor(k,z))^0.33
    fReturn=(k::Float64,z::Int64)->exp(gridTfp[z])*0.33*(fLabor(k,z)/fCapital(k,z))^(0.67)
    fOutput=(k::Float64,z::Int64)->exp(gridTfp[z])*k^0.33*fLabor(k,z)^0.67



    return fCapital,fConsumption,fOutput,fLabor,fWage,fReturn,fValue  

end



























