################################
###Guvenen 2007 Replication#####
################################
using LinearAlgebra
using Distributions
using Random
using Plots




##################################################################
###Define all structs at the top level--struct for all funtions###
##################################################################

struct Parameters
    σ_α
    σ_β
    σ_αβ
    λ
    β_bar
    α_bar
    CRRA
    δ
    ρ
    σ_η
    σ_ϵ
    T
    TStar
    Pb
end


##############################
###Structs for grid builder###
##############################

struct PanelObservation
    y
    α_hat
    β_hat
    z_hat
    η
    ϵ
    z
    Σ_hat
end


struct xObservation
    α_hat
    β_hat
    z_hat 
end


struct ΣObservation
    Σ_hat
end


struct xGridPoint
    α_hat
    β_hat
    z_hat 
end


struct GridPoint
    ω
    y
    α_hat
    β_hat 
    z_hat
end


struct PeriodGrid
    grid::Array{GridPoint,1}
end

struct LearningPaths

    αLearningPath::Array{Float64,1}
    βLearningPath::Array{Float64,1}
    αGapPath::Array{Float64,1}
    βGapPath::Array{Float64,1}
    
end

GuvenenAll=Parameters(0.022,0.00038,-0.0020,0,0.0009,1.5,2.0,0.966,0.821,0.029,0.047,40,95,0.96)

### To generate plots, enter "main(GuvenenAll)" in the REPL##

function main(parameterObject)
    tic=@elapsed begin

    ###Rename Parameters###
    σ_α=parameterObject.σ_α
    σ_β=parameterObject.σ_β
    σ_αβ=parameterObject.σ_αβ
    λ=parameterObject.λ
    β_bar=parameterObject.β_bar
    α_bar=parameterObject.α_bar
    ϕ=parameterObject.CRRA
    δ=parameterObject.δ
    ρ=parameterObject.ρ
    σ_η=parameterObject.σ_η
    σ_ϵ=parameterObject.σ_ϵ
    T=parameterObject.T
    TStar=parameterObject.TStar
    Pb=parameterObject.Pb
    CRRA=parameterObject.CRRA
    

    ###This program inputs parameters and returns simulated sequences of 
    ###consumption and savings for a cross-section of idiosyncratic paramters###

    function GridBuilder(II,T,J,yGridSize,wealthGridSize)
        ###Constituent functions: panel samples generator, Kalman filter, 
        ###grid construction###


        function Panels()

            ηArray=zeros(T,II,J)
            ϵArray=zeros(T,II,J)
            αβCross=zeros(2,II)
            zArray=zeros(T,II,J)
            yArray=zeros(T,II,J)

            rand!(MvNormal([α_bar;β_bar],[σ_α σ_αβ; σ_αβ σ_β]),αβCross)


            for j=1:J

                for i=1:II

                    ηArray[1,i,j]=rand(Normal(0,sqrt(σ_η)))
                    ϵArray[1,i,j]=rand(Normal(0,sqrt(σ_ϵ)))
                    zArray[1,i,j]=ηArray[1,i,j]
                    yArray[1,i,j]=αβCross[1,i]+αβCross[2,i]+zArray[1,i,j]+ϵArray[1,i,j]

                    for t=2:T

                    ηArray[t,i,j]=rand(Normal(0,sqrt(σ_η)))
                    ϵArray[t,i,j]=rand(Normal(0,sqrt(σ_ϵ)))
                    zArray[t,i,j]=ρ*zArray[t-1,i,j]+ηArray[t,i,j]
                    yArray[t,i,j]=αβCross[1,i]+αβCross[2,i]*t+zArray[t,i,j]+ϵArray[t,i,j]

                    end

                end

            end

            return ηArray,ϵArray,zArray,yArray,αβCross

        end

        statePanel=Panels()
        ηArray=statePanel[1]
        ϵArray=statePanel[2]
        zArray=statePanel[3]
        yArray=statePanel[4]
        αβCross=statePanel[5]
        
        
        ###Kalman Filter function: takes a single time path and returns
        ### state forecasts over the time path###

        function KalmanFilter(ySeries)
            
            ###x is vector of filtered state forecasts of α, β, and z###
            ###Generate state-space representation coefficients###
            A=[1 0 0;0 1 0;0 0 ρ]
            C=[0;0;σ_η]
            xPrior=[α_bar;β_bar;0]
            ΣPrior=[σ_α σ_αβ 0; σ_αβ (1-λ)*σ_β 0; 0 0 σ_η]

            x=zeros(3,T)
            Σ=zeros(3,3,T)
            x[:,1]=xPrior
            Σ[:,:,1]=ΣPrior

            ###Kalman recursion###
            for t=1:T-1
                
                G=convert(Matrix{Float64},[1 t 1])
                a=ySeries[t].-G*x[:,t]
                K=A*Σ[:,:,t]*G'*(G*Σ[:,:,t]*G'.+σ_ϵ)^(-1)
                x[:,t+1]=A*x[:,t].+K*a
                Σ[:,:,t+1]=C*C'.+K*σ_ϵ*K'.+(A-K*G)*Σ[:,:,t]*(A-K*G)'

                
            end

            return x,Σ

        end

        ###Note for panel sample organization--the panel sample should be a 3-dimensional
        ###array of objects, each one of which has a y,αhat,βhat,zhat,η,ϵ###


        PanelSample=Array{PanelObservation,3}(undef,T,II,J)
        
        for i=1:II
            
            for j=1:J

                xSample,ΣSample=KalmanFilter(yArray[:,j,i])

                for t=1:T

                    PanelSample[t,i,j]=PanelObservation(yArray[t,i,j],xSample[1,t],xSample[2,t],
                    xSample[3,t],ηArray[t,i,j],ϵArray[t,i,j],zArray[t,i,j],ΣSample[:,:,t])

                end

            end

        end


        ###PeriodGrid builds the grid over (w,y,αhat,βhat,zhat) for each t###
        ###xCross is a cross section 2 (IxJ) dimensional array of objects αhat,βhat,zhat

        xPanel=Array{xObservation,3}(undef,T,II,J)
        ΣPanel=Array{ΣObservation,3}(undef,T,II,J)

        for i=1:II
            for j=1:J
                for t=1:T
                    xPanel[t,i,j]=xObservation(PanelSample[t,i,j].α_hat,PanelSample[t,i,j].β_hat,
                    PanelSample[t,i,j].z_hat)

                    ΣPanel[t,i,j]=ΣObservation(PanelSample[t,i,j].Σ_hat)
                end
            end
        end
        

        function PeriodGridBuilder(xCross,ΣCross,t)

            ###Find the bounds of grid in each dimension###
            αlist=zeros(II*J)
            βlist=zeros(II*J)
            zlist=zeros(II*J)

            for i=1:II

                for j=1:J
                
                    αlist[(i-1)*j+j]=xCross[i,j].α_hat
                    βlist[(i-1)*j+j]=xCross[i,j].β_hat
                    zlist[(i-1)*j+j]=xCross[i,j].z_hat

                end

            end

            αGridBounds=[minimum(αlist),maximum(αlist)]
            βGridBounds=[minimum(βlist),maximum(βlist)]
            zGridBounds=[minimum(zlist),maximum(zlist)]

            αPreliminary=zeros(21)
            βPreliminary=zeros(21)
            zPreliminary=zeros(21)
            
            for i=1:21

                αPreliminary[i]=αGridBounds[1]+((αGridBounds[2]-αGridBounds[1])/20)*(i-1)
                βPreliminary[i]=βGridBounds[1]+((βGridBounds[2]-βGridBounds[1])/20)*(i-1)
                zPreliminary[i]=zGridBounds[1]+((zGridBounds[2]-zGridBounds[1])/20)*(i-1)

            end
            
            ###Create preliminary grid--21x21x21 and count population###
            cubeCounter=zeros(20,20,20)
            cubeInhabitant=Array{Int64,2}(undef,2,0)

            for αi=1:20

                for βi=1:20

                    for zi=1:20

                        for i=1:II

                            for j=1:J

                                b1=xCross[i,j].α_hat≥αPreliminary[αi]
                                b2=xCross[i,j].α_hat≤αPreliminary[αi+1]
                                b3=xCross[i,j].β_hat≥βPreliminary[βi]
                                b4=xCross[i,j].β_hat≤βPreliminary[αi+1]
                                b5=xCross[i,j].z_hat≥zPreliminary[zi]
                                b6=xCross[i,j].z_hat≤zPreliminary[zi+1]

                                if b1&&b2&&b3&&b4&&b5&&b6

                                    cubeCounter[αi,βi,zi]=cubeCounter[αi,βi,zi]+1
                                    if cubeCounter[αi,βi,zi]==1
                                        cubeInhabitant=hcat(cubeInhabitant,[i;j])
                                    end

                                end

                            end
                            
                        end

                    end

                end

            end


            ###Use the nonempty preliminary cubes to build final grid for x states###

            xGridFinal=Array{xGridPoint,1}(undef,0)

            for αi=1:20

                for βi=1:20

                    for zi=1:20

                        if cubeCounter[αi,βi,zi]>0

                            push!(xGridFinal,xGridPoint((αPreliminary[αi]+αPreliminary[αi+1])/2,
                            (βPreliminary[βi]+βPreliminary[βi+1])/2,
                            (zPreliminary[zi]+zPreliminary[zi+1]/2)))

                        end

                    end

                end

            end


            ###Now add the y state###

            gridFinalSize=length(xGridFinal)*wealthGridSize*yGridSize
            gridFinal=Array{GridPoint,1}(undef,gridFinalSize)

            for q=1:length(xGridFinal)

                G=[1 t 1]
                xGridVec=[xGridFinal[q].α_hat;xGridFinal[q].β_hat;xGridFinal[q].z_hat]
                ΣStep=ΣCross[cubeInhabitant[1,q],cubeInhabitant[2,q]].Σ_hat
                yGridBound=[exp(G*xGridVec-3*(G*ΣStep*G'.+σ_ϵ));exp(G*xGridVec+3*(G*ΣStep*G'.+σ_ϵ))]
                wealthGridBound=[0;yGridBound[2]*5]

                yGridLoad=zeros(yGridSize)
                wealthGridLoad=zeros(wealthGridSize)
                for i=1:yGridSize

                    yGridLoad[i]=yGridBound[1]+((yGridBound[2]-yGridBound[1])/(yGridSize-1))*(i-1)

                    for j=1:wealthGridSize
                        wealthGridLoad[j]=wealthGridBound[1]+((wealthGridBound[2]-wealthGridBound[1])/
                        wealthGridSize-1)*(j-1)

                        gridFinalIndex=(q-1)*yGridSize*wealthGridSize+(i-1)*wealthGridSize+j
                        gridFinal[gridFinalIndex]=GridPoint(wealthGridLoad[j],yGridLoad[i],
                        xGridFinal[q].α_hat,xGridFinal[q].β_hat,xGridFinal[q].z_hat)

                    end

                end              

            end

            return gridFinal

        end

        ###Last step--use PeriodGridBuilder to build a grid for each t###

        gridFull=Array{PeriodGrid}(undef,T)
        for t=1:T
            
            gridFull[t]=PeriodGrid(PeriodGridBuilder(xPanel[t,:,:],ΣPanel[t,:,:],t))
            println(t)

        end

        return gridFull,PanelSample,αβCross


    end

    gridFull,panelSample,αβCross=GridBuilder(100,T,100,8,15)

    T=size(panelSample)[1]
    II=size(panelSample)[2]
    J=size(panelSample)[3]

    end

    println(tic)

    function Rowenhorst(N,δ,σ2,μ)

        stateValues=zeros(N)
        stateValues[1]=μ-((σ2*(N-1))^(1/2))/(1-δ^2)^(1/2)
        stateValues[N]=μ+((σ2*(N-1))^(1/2))/(1-δ^2)^(1/2)
    
        for j=2:N-1
    
            stateValues[j]=stateValues[j-1]+(2/(N-1))*((σ2*(N-1))^(1/2))/(1-δ^2)^(1/2)
            
        end
    
        p=(1+δ)/2
        T=[p 1-p;1-p p]
    
        for j=2:N-1
    
            vect1=zeros(j,1)
            vect2=zeros(1,j+1)
    
            T0=hcat(T,vect1)
            T0=vcat(T0,vect2)
    
            T1=hcat(vect1,T)
            T1=vcat(T1,vect2)
    
            T2=hcat(T,vect1)
            T2=vcat(vect2,T2)
    
            T3=hcat(vect1,T)
            T3=vcat(vect2,T3)
    
            T_tilde=p*T0+(1-p)*T1+(1-p)*T2+p*T3
            inner=0.5*ones(j-1)
            iden=diagm(vcat([1],inner,[1]))
    
            T=iden*T_tilde
    
    
        end

    



    end

    function LearningPathGenerate(panelPath::Array{PanelObservation},αβ,T)

        σ_αPath=zeros(T)
        σ_βPath=zeros(T)
        αGapPath=zeros(T)
        βGapPath=zeros(T)
        αLearningPath=zeros(T)
        βLearningPath=zeros(T)

        for t=1:T

            σ_αPath[t]=panelPath[t].Σ_hat[1,1]
            σ_βPath[t]=panelPath[t].Σ_hat[2,2]
            αGapPath[t]=log(abs(αβ[1]/panelPath[t].α_hat))
            βGapPath[t]=log(abs(αβ[2]/panelPath[t].β_hat))

        end

        for t=2:T

            αLearningPath[t]=log(1/σ_αPath[t])-log(1/σ_αPath[t-1])
            βLearningPath[t]=log(1/σ_βPath[t])-log(1/σ_βPath[t-1])

        end

        return LearningPaths(αLearningPath,βLearningPath,αGapPath,βGapPath)

    end


    function PathMean(panel)

        T=size(panel)[1]
        II=size(panel)[2]
        J=size(panel)[3]
        averageSeries=zeros(T)
        
        for t=1:T

            averageSeries[t]=mean(panel[t,:,:])

        end

        return averageSeries

    end

    learningPathsPanel=Array{LearningPaths,2}(undef,II,J)
    
    for j=1:J

        for i=1:II

            learningPathsPanel[i,j]=LearningPathGenerate(panelSample[:,i,j],αβCross[:,i],T)

        end

    end

    αLearningPanel=Array{Float64,3}(undef,T,II,J)
    βLearningPanel=Array{Float64,3}(undef,T,II,J)
    αGapPanel=Array{Float64,3}(undef,T,II,J)
    βGapPanel=Array{Float64,3}(undef,T,II,J)

    for i=1:II

        for j=1:J

            αLearningPanel[:,i,j]=learningPathsPanel[i,j].αLearningPath
            βLearningPanel[:,i,j]=learningPathsPanel[i,j].βLearningPath
            αGapPanel[:,i,j]=learningPathsPanel[i,j].αGapPath
            βGapPanel[:,i,j]=learningPathsPanel[i,j].βGapPath

        end

    end

    αLearningMean=PathMean(αLearningPanel)
    βLearningMean=PathMean(βLearningPanel)
    αGapPath=PathMean(αGapPanel)
    βGapPath=PathMean(βGapPanel)

    display(plot(26:1:64,αLearningMean[2:40],title="Change in the precision of beliefs about α",xlabel="Age",
    ylabel="log(1/σα_t+1|t)-log(1/σα_t)"))

    display(plot(26:1:64,βLearningMean[2:40],title="Change in the precision of beliefs about β",xlabel="Age",
    ylabel="log(1/σβ_t+1|t)-log(1/σβ_t|t-1)"))

    display(plot(25:1:64,αGapPath,title="log-difference gap between α and filtered forecast"
    ,xlabel="Age",ylabel="log-difference"))

    display(plot(25:1:64,βGapPath,title="log-difference gap between β and filtered forecast"
    ,xlabel="Age",ylabel="log-difference"))

    return αLearningMean,βLearningMean,αGapPath,βGapPath,panelSample




end

