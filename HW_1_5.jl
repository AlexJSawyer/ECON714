###########################
###Set global parameters###
###########################

###n is the number of households and m is the number of goods;
#please set these equal to each other###
n=3
m=3

α=(1,2,3)
ωAggregate=-1.5*ones(n,m)
#for j=1:m
 #   ωAggregate[j,j]=-3
#end
eAggregate=ones(n,m)
#for j=1:m
 #   eAggregate[j,j]=2
#end
eAggregate[:,3]=2*ones(n,1)
tol=10^(-7)

####################################################################
###The below solves the household consumption bundle given prices###
####################################################################

function HouseholdSolver(p,α,ω,e,tol)


    ###Defining the function to be zeroed and auxillary functions###
    function F(x,p,α,ω,e)
        
        y=zeros(m,1)
        sum=e[1]
        for i=2:m
            sum=sum+p[i]*(e[i]-x[i])
        end

        for i=1:m
            y[i]=x[i]-((α[1]/α[i])*p[i]*sum^(ω[i]))^(1/ω[i])
        end

        return y

    end

    function f(x,p,α,ω,e)
        return (1/2)*F(x,p,α,ω,e)'*F(x,p,α,ω,e)
    end

    function CenteredDifference(x,p,F)
        h=10^(-6)
        M=size(x)[1]
        N=size(F(x,p,α,ω,e))[1]
        Df=zeros(N,M)
        forwardPoint=zeros(M,1)
        backwardPoint=zeros(M,1)
    
        for j=1:M
            for i=1:N
                mStep=zeros(M,1)
                mStep[j,1]=h
                forwardPoint=x+mStep
                backwardPoint=x-mStep
                Df[i,j]=(F(forwardPoint,p,α,ω,e)[i]-F(backwardPoint,p,α,ω,e)[i])/(2*h)
            end
        end
    
        return Df
    
    end
    

    ###Main body of root-finder###

    ###Initialize first guesses###
    xGuess=ones(m,1)
    FGuess=F(xGuess,p,α,ω,e)
    fGuess=f(xGuess,p,α,ω,e)
    J=CenteredDifference(xGuess,p,F)
    Δf=FGuess'*J

    β=10^(-4)

    ###Convergence loop###
    while fGuess[1,1]>tol
        ###Compute Newton direction and try full step###
        q=-J^(-1)*FGuess
        if only(f(xGuess+q,p,α,ω,e))≤only(f(xGuess,p,α,ω,e)+β*Δf*q)
            xGuess=xGuess+q
        else
            ###Try the first backtrack###
            function g(λ)
                return f(xGuess+λ*q,p,α,ω,e)
            end
            function Dg(λ)
                return F(xGuess+λ*q,p,α,ω,e)'*CenteredDifference(xGuess+λ*q,p,F)*q
            end

            λ=-Dg(0)/(2*(g(1)-g(0)-Dg(0)))
            if only(f(xGuess+λ*q,p,α,ω,e))≤only(f(xGuess,p,α,ω,e)+β*Δf*λ*q)
                xGuess=xGuess+λ*q
            else
                ###Try the second and subsequent backtracks###

                ###Intialize the subsequent backtrack interation
                matrix1=[1/λ^2 -1;-1/λ^2 λ]
                matrix2=[g(λ)-Dg(0)*λ-g(0);g(1)-Dg(0)-g(0)]
                (a,b)=(1/(λ-1))*matrix1*matrix2
                λ1=(-b+sqrt(b^2-3*a*Dg(0)))/(3*a)
                λ2=λ
                
                ###Subsequent backtrack interation###
                while only(f(xGuess+λ1*q))≥only(f(xGuess)+β*Δf*λ1*q)
                    matrix1=[1/λ1^2 -1/λ2^2; -λ2/λ1 λ1/λ2^2]
                    matrix2=[g(λ1)-Dg(0)*λ1-g(0);g(λ2)-Dg(0)*λ2-g(0)]

                    (a,b)=(1/(λ1-λ2))*matrix1*matrix2
                    λ2=λ1
                    λ1=(-b+sqrt(b^2-3*a*Dg(0)))/(3*a)
                    λ1=max(0.1*λ2,λ1)
                    λ1=min(0.5*λ2,λ1)
                end

                xGuess=xGuess+λ1*q

            end

        end

        FGuess=F(xGuess,p,α,ω,e)
        fGuess=f(xGuess,p,α,ω,e)
        J=CenteredDifference(xGuess,p,F)
        Δf=FGuess'*J

    end

    return xGuess

end

##################################################################
###This is the main function, solves for market-clearing prices###
##################################################################

function MarketClearing(α,ω,e,tol)
    tic=@elapsed begin

    ###Define excess demand function and auxillary functions
    function H(p)

        ###The input of the function is the price vector;
        ###demand has to be found for each household given p###

        x=zeros(n,m)
        for i=1:n
            x[i,:]=HouseholdSolver(p,α,ω[i,:],e[i,:],tol)'
        end

        y=zeros(m,1)
        
        for j=1:m
            sum=0
            for i=1:n
                sum=sum+e[i,j]-x[i,j]
            end
            y[j]=sum
        end
        return y
    end

    function h(p)
        return (1/2)*H(p)'*H(p)
    end

    function CenteredDifference(p)

        δ=10^(-6)
        N=size(H(p))[1]
        M=size(p)[1]
        Dh=zeros(N,M)
        forwardPoint=zeros(M,1)
        backwardPoint=zeros(M,1)

        for j=1:M
            for i=1:N
                mStep=zeros(M,1)
                mStep[j,1]=δ
                forwardPoint=p+mStep
                backwardPoint=p-mStep
                Dh[i,j]=(H(forwardPoint)[i]-H(backwardPoint)[i])/(2*δ)
            end
        end

        return Dh
    end

    ###Main body of root-finder###

    ###Initialize first guesses###
    pGuess=ones(m,1)
    for i=1:m
        pGuess[i]=1/sum(e[:,i])
    end
    pGuess=pGuess/pGuess[1]
    HGuess=H(pGuess)
    hGuess=h(pGuess)
    J=CenteredDifference(pGuess)
    Δh=HGuess'*J

    β=10^(-4)

    counter=0
    ###Convergence loop###
    while hGuess[1,1]>tol
        ###Compute Newton direction and try full step###
        q=-J^(-1)*HGuess
        if only(h(pGuess+q))≤only(h(pGuess)+β*Δh*q)
            pGuess=pGuess+q
        else
            ###Try the first backtrack###
            function g(λ)
                return only(h(pGuess+λ*q))
            end
            function Dg(λ)
                return only(H(pGuess+λ*q)'*CenteredDifference(pGuess+λ*q)*q)
            end

            λ=-Dg(0)/(2*(g(1)-g(0)-Dg(0)))
            if only(h(pGuess+λ*q))≤only(h(pGuess)+β*Δh*λ*q)
                pGuess=pGuess+λ*q
            else
                ###Try the second and subsequent backtracks###

                ###Intialize the subsequent backtrack interation
                matrix1=[1/λ^2 -1;-1/λ^2 λ]
                matrix2=[g(λ)-Dg(0)*λ-g(0);g(1)-Dg(0)-g(0)]
                (a,b)=(1/(λ-1))*matrix1*matrix2
                λ1=(-b+sqrt(b^2-3*a*Dg(0)))/(3*a)
                λ2=λ
                
                ###Subsequent backtrack interation###
                while only(h(pGuess+λ1*q))≥only(h(pGuess)+β*Δh*λ1*q)
                    matrix1=[1/λ1^2 -1/λ2^2; -λ2/λ1 λ1/λ2^2]
                    matrix2=[g(λ1)-Dg(0)*λ1-g(0);g(λ2)-Dg(0)*λ2-g(0)]

                    (a,b)=(1/(λ1-λ2))*matrix1*matrix2
                    λ2=λ1
                    λ1=(-b+sqrt(b^2-3*a*Dg(0)))/(3*a)
                    λ1=max(0.1*λ2,λ1)
                    λ1=min(0.5*λ2,λ1)
                end

                pGuess=pGuess+λ1*q

            end

        end

        HGuess=H(pGuess)
        hGuess=h(pGuess)
        J=CenteredDifference(pGuess)
        Δp=HGuess'*J

        counter=counter+1
        println(counter)
    end

    end
    return pGuess,tic

end