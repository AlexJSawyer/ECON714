##################################################
### Comparision of global methods to solve RBC ###
### Alexander Sawyer                           ###
### Fall 2021                                  ###
##################################################

include("HW_2_VFI.jl")
include("HW_2_projection.jl")
include("HW_2_FE.jl")
using Plots


function main()

gridTfp=[-0.05,0,0.05]
ΠTpf=[0.97 0.03 0;0.01 0.98 0.01;0 0.03 0.97]
ΠTfp=ΠTpf

###Time each method ###
tocVFI=@elapsed begin
    fVFI=Multigrid(2,2,1,2.5)
end

tocVFIAccel=@elapsed begin
    fVFIAccel=Multigrid(2,2,0.05,2.5)
end

tocVFIMulti=@elapsed begin
    fVFIMulti=Multigrid(2,4,0.05,1)
end

tocProject=@elapsed begin
    fProject=ProjectedCheby(5)
end
println("projection done")
tocFE=@elapsed begin
    fFE=FunctionBuild(8)
end

### Plot policy functions ###

capitalVFI=fVFI[1]
capitalVFIAccel=fVFIAccel[1]
capitalVFIMulti=fVFIMulti[1]
capitalProject=fProject[1]
capitalFE=fFE[1]

display(plot(gridCapitalMin:0.01:gridCapitalMax,[y->capitalVFI(y,2),y->capitalVFIAccel(y,2),
y->capitalVFIMulti(y,2),y->capitalProject(y,2)],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Capital Policy Function",legend=:topleft))

function Recession(evolutionLaws)
   
    zTimePath=Array{Int64,1}(undef,50)
    cTimePath=zeros(50)
    kTimePath=zeros(50)
    yTimePath=zeros(50)
    lTimePath=zeros(50)
    wTimePath=zeros(50)
    rTimePath=zeros(50)
    kNextTimePath=zeros(51)

    for t=1:33
        zTimePath[t]=3
    end

    for t=34:50
        zTimePath[t]=2
    end

    kNextTimePath[1]=capitalSS

    for t=1:50

        kTimePath[t]=kNextTimePath[t]
        kNextTimePath[t+1]=evolutionLaws[1](kTimePath[t],zTimePath[t])
        cTimePath[t]=evolutionLaws[2](kTimePath[t],zTimePath[t])
        yTimePath[t]=evolutionLaws[3](kTimePath[t],zTimePath[t])
        lTimePath[t]=evolutionLaws[4](kTimePath[t],zTimePath[t])
        wTimePath[t]=evolutionLaws[5](kTimePath[t],zTimePath[t])
        rTimePath[t]=evolutionLaws[6](kTimePath[t],zTimePath[t])

    end

    cLogPath=zeros(50)
    kLogPath=zeros(50)
    yLogPath=zeros(50)
    lLogPath=zeros(50)
    wLogPath=zeros(50)

    for t=1:50

        cLogPath[t]=log(cTimePath[t])-log(cTimePath[1])
        kLogPath[t]=log(kTimePath[t])-log(kTimePath[1])
        yLogPath[t]=log(yTimePath[t])-log(yTimePath[1])
        lLogPath[t]=log(lTimePath[t])-log(lTimePath[1])
        wLogPath[t]=log(wTimePath[t])-log(wTimePath[1])

    end

    return kLogPath,cLogPath,yLogPath,lLogPath,wLogPath,rTimePath

end

VFIPaths=Recession(fVFI)
VFIAccelPaths=Recession(fVFIAccel)
VFIMultiPaths=Recession(fVFIMulti)
ProjectPaths=Recession(fProject)
#FEPaths=Recession(fFE)

display(plot(1:1:50,[VFIPaths[1],VFIAccelPaths[1],VFIMultiPaths[1],ProjectPaths[1]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of capital to negative TFP shock"))

display(plot(1:1:50,[VFIPaths[2],VFIAccelPaths[2],VFIMultiPaths[2],ProjectPaths[2]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of consumption to negative TFP shock"))

display(plot(1:1:50,[VFIPaths[3],VFIAccelPaths[3],VFIMultiPaths[3],ProjectPaths[3]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of output to negative TFP shock"))

display(plot(1:1:50,[VFIPaths[4],VFIAccelPaths[4],VFIMultiPaths[4],ProjectPaths[4]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of labor to negative TFP shock"))

display(plot(1:1:50,[VFIPaths[5],VFIAccelPaths[5],VFIMultiPaths[5],ProjectPaths[5]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of wages to negative TFP shock"))

display(plot(1:1:50,[VFIPaths[6],VFIAccelPaths[6],VFIMultiPaths[6],ProjectPaths[6]],
label=["VFI" "VFI Accelerator" "VFI Multigrid" "Chebyshev"],
title="Response of return rate to negative TFP shock"))

println([tocVFI tocVFIAccel tocVFIMulti tocProject tocFE])


end