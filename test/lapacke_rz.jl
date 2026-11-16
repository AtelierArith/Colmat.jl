using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32,Float64,ComplexF32,ComplexF64),J in (Int32,Int64)
    @eval function lapacke288_rz_factor(::Type{$T},::Type{$J},p,high,layout,m,n,a,lda,tau,work,lwork,name,route)
        lapacke288_selected_entry(name,$J,route) do
            if high
                ccall(p,$J,(Cint,$J,$J,Ptr{$T},$J,Ptr{$T}),layout,m,n,a,lda,tau)
            else
                ccall(p,$J,(Cint,$J,$J,Ptr{$T},$J,Ptr{$T},Ptr{$T},$J),layout,m,n,a,lda,tau,work,lwork)
            end
        end
    end
    @eval function lapacke288_rz_apply(::Type{$T},::Type{$J},p,high,layout,side,tr,m,n,k,l,a,lda,tau,c,ldc,work,lwork,name,route)
        lapacke288_selected_entry(name,$J,route) do
            if high
                ccall(p,$J,(Cint,Int8,Int8,$J,$J,$J,$J,Ptr{$T},$J,Ptr{$T},Ptr{$T},$J),layout,Int8(side),Int8(tr),m,n,k,l,a,lda,tau,c,ldc)
            else
                ccall(p,$J,(Cint,Int8,Int8,$J,$J,$J,$J,Ptr{$T},$J,Ptr{$T},Ptr{$T},$J,Ptr{$T},$J),layout,Int8(side),Int8(tr),m,n,k,l,a,lda,tau,c,ldc,work,lwork)
            end
        end
    end
end
for T in (Float32,Float64,ComplexF32,ComplexF64)
    @eval function lapacke288_rz_reference_factor(::Type{$T},p,original)
        m,n=size(original);la=m+2
        a=fill($T(NaN),la,n)
        for j in 1:n,i in 1:min(m,j);a[i,j]=original[i,j];end
        saved=copy(a);tau=fill($T(778),m+2);work=fill($T(779),max(1,m)+2);info=fill(Int64(-777),3)
        ccall(p,Cvoid,(Ref{Int64},Ref{Int64},Ptr{$T},Ref{Int64},Ptr{$T},Ptr{$T},Ref{Int64},Ptr{Int64}),m,n,a,la,view(tau,2:m+1),view(work,2:length(work)-1),max(1,m),view(info,2:2))
        @test info==[-777,0,-777]
        @test tau[[1,end]]==fill($T(778),2)&&work[[1,end]]==fill($T(779),2)
        @test isequal(a[m+1:end,:],saved[m+1:end,:])
        for j in 1:min(m,n),i in j+1:m;@test isequal(a[i,j],saved[i,j]);end
        (;a=a[1:m,:],tau=tau[2:m+1])
    end
    @eval function lapacke288_rz_fortran_apply(::Type{$T},p,side,tr,a0,tau0,c0,k,l;external=true)
        m,n=size(c0);order=side=='L' ? m : n
        la=k+2;lc=m+2
        a=zeros($T,la,order)
        # Only the final L columns carry reflector data. Avoid unused prefixes.
        for i in 1:k,j in order-l+1:order;a[i,j]=a0[i,j];end
        tau=fill($T(778),k+2);tau[2:k+1].=tau0[1:k]
        c=fill($T(779),lc,n);c[1:m,:].=c0
        sa,st=copy(a),copy(tau)
        lw=max(1,side=='L' ? n : m);work=fill($T(780),lw+2);info=fill(Int64(-777),3)
        if external
            ccall(p,Cvoid,(Ref{UInt8},Ref{UInt8},Ref{Int64},Ref{Int64},Ref{Int64},Ref{Int64},Ptr{$T},Ref{Int64},Ptr{$T},Ptr{$T},Ref{Int64},Ptr{$T},Ref{Int64},Ptr{Int64},Csize_t,Csize_t),UInt8(side),UInt8(tr),m,n,k,l,a,la,view(tau,2:k+1),c,lc,view(work,2:lw+1),lw,view(info,2:2),1,1)
        else
            ccall(p,Cvoid,(Ref{UInt8},Ref{UInt8},Ref{Int64},Ref{Int64},Ref{Int64},Ref{Int64},Ptr{$T},Ref{Int64},Ptr{$T},Ptr{$T},Ref{Int64},Ptr{$T},Ref{Int64},Ptr{Int64}),UInt8(side),UInt8(tr),m,n,k,l,a,la,view(tau,2:k+1),c,lc,view(work,2:lw+1),lw,view(info,2:2))
        end
        @test info==[-777,0,-777]
        @test a==sa&&tau==st
        @test all(==($T(779)),c[m+1:end,:])&&work[[1,end]]==fill($T(780),2)
        c[1:m,:]
    end
end
@testset "LAPACKE RZ factor storage and independent Z consumers" begin
    native=Libdl.dlopen(Colmat.backend_library());trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(only(filter(l->l.interface==:ilp64&&occursin("openblas",lowercase(l.libname)),BLAS.get_config().loaded_libs)).libname)
    for (T,prefix) in ((Float32,"s"),(Float64,"d"),(ComplexF32,"c"),(ComplexF64,"z")),J in (Int32,Int64),
        (m,n) in ((0,0),(0,3),(1,1),(2,2),(1,4),(2,5),(3,7)),deficient in (false,true),exponent in (-1,0,1)
        R=real(T);tol=R(4000)*eps(R);l=n-m
        original=T[i<=j ? cos(i+2j)+(i==j ? 2 : 0) : 0 for i in 1:m,j in 1:n]
        if T<:Complex
            for i in 1:m,j in i:n;original[i,j]+=T(im)*sin(2i+j)/5;end
        end
        deficient&&m>0&&(original[m,:].=0)
        original .*= R(2)^(exponent*(R==Float32 ? 40 : 400))
        fp=Libdl.dlsym(reference,prefix*"tzrzf_64_");family=T<:Complex ? "unmrz" : "ormrz"
        gp=Libdl.dlsym(reference,prefix*family*"_64_");op=Libdl.dlsym(native,prefix*family*"_64_")
        @test fp!=Libdl.dlsym(native,prefix*"tzrzf_64_")&&gp!=op
        ext=lapacke288_rz_reference_factor(T,fp,original)
        rz=lapacke288_rz_fortran_apply(T,op,'R','N',ext.a,ext.tau,Matrix{T}(I,n,n),m,l;external=false)
        denom=max(norm(original),floatmin(R))
        @test norm(hcat(triu(ext.a[:,1:m]),zeros(T,m,l))*rz-original)/denom<=tol*20
        @test rz'*rz≈Matrix{T}(I,n,n) atol=tol
        for route in (:dll,:lbt_trampoline),layout in (101,102),high in (false,true)
            name="LAPACKE_"*prefix*"tzrzf"*(high ? "" : "_work");suffix=J==Int64 ? "64_" : ""
            p=Libdl.dlsym(route==:dll ? native : trampoline,name*suffix)
            @test BLAS.lbt_get_forward(name,J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(native,name*suffix)
            la=layout==101 ? n+2 : m+2
            ai(i,j)=layout==101 ? (i-1)*la+j : i+(j-1)*la
            a=fill(T(NaN),layout==101 ? max(1,m)*la : max(1,n)*la);mask=falses(length(a))
            for i in 1:m,j in 1:n
                if high||i<=j;a[ai(i,j)]=original[i,j];end
                mask[ai(i,j)]=i<=j
            end
            saved=copy(a);tau=fill(T(778),m+2);minimum=m==n ? 1 : max(1,m);w=fill(T(779),max(1,m)+2)
            call(lw)=lapacke288_rz_factor(T,J,p,high,layout,m,n,a,la,view(tau,2:m+1),view(w,2:length(w)-1),lw,name,route)
            if !high
                @test call(-1)==0
                @test isequal(a,saved)&&all(==(T(778)),tau)
                @test isreal(w[2])&&real(w[2])>=minimum
                if m==0||m==n;@test w[2]==one(T);end
            end
            @test call(minimum)==0
            @test isequal(a[.!mask],saved[.!mask])
            @test tau[[1,end]]==fill(T(778),2)&&w[[1,end]]==fill(T(779),2)
            if !high&&(m==0||m==n);@test w[2]==one(T);end
            own=T[a[ai(i,j)] for i in 1:m,j in 1:n]
            z=lapacke288_rz_fortran_apply(T,gp,'R','N',own,tau[2:m+1],Matrix{T}(I,n,n),m,l)
            @test norm(hcat(triu(own[:,1:m]),zeros(T,m,l))*z-original)/denom<=tol*20
            @test z'*z≈Matrix{T}(I,n,n) atol=tol
        end
    end
end
@testset "LAPACKE RZ application: independent factors, sides and conjugation" begin
    native=Libdl.dlopen(Colmat.backend_library());trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(only(filter(l->l.interface==:ilp64&&occursin("openblas",lowercase(l.libname)),BLAS.get_config().loaded_libs)).libname)
    for (T,prefix) in ((Float32,"s"),(Float64,"d"),(ComplexF32,"c"),(ComplexF64,"z")),J in (Int32,Int64),
        (rows,order) in ((0,0),(0,3),(1,1),(2,5),(3,7)),side in ('L','R'),tr in (T<:Complex ? ('N','C') : ('N','T'))
        R=real(T);tol=R(4000)*eps(R)
        original=T[i<=j ? cos(i+2j)+(i==j ? 2 : 0) : 0 for i in 1:rows,j in 1:order]
        if T<:Complex
            for i in 1:rows,j in i:order;original[i,j]+=T(im)*sin(2i+j)/5;end
        end
        ext=lapacke288_rz_reference_factor(T,Libdl.dlsym(reference,prefix*"tzrzf_64_"),original)
        family=T<:Complex ? "unmrz" : "ormrz";gp=Libdl.dlsym(reference,prefix*family*"_64_")
        for k in unique((0,min(1,rows),rows)),l in unique((0,(order-rows)÷2,order-rows))
            m,n=side=='L' ? (order,3) : (3,order)
            c0=T[sin(i+3j) for i in 1:m,j in 1:n]
            T<:Complex&&(c0 .+= T(im).*T[cos(3i+j)/7 for i in 1:m,j in 1:n])
            expected=lapacke288_rz_fortran_apply(T,gp,side,tr,ext.a,ext.tau,c0,k,l)
            z=lapacke288_rz_fortran_apply(T,gp,'R','N',ext.a,ext.tau,Matrix{T}(I,order,order),k,l)
            operator=tr=='N' ? z : z'
            @test expected≈(side=='L' ? operator*c0 : c0*operator) atol=tol
            for route in (:dll,:lbt_trampoline),layout in (101,102),high in (false,true)
                name="LAPACKE_"*prefix*family*(high ? "" : "_work");suffix=J==Int64 ? "64_" : ""
                p=Libdl.dlsym(route==:dll ? native : trampoline,name*suffix)
                @test BLAS.lbt_get_forward(name,J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(native,name*suffix)
                la=layout==101 ? order+2 : k+2;lc=layout==101 ? n+2 : m+2
                ai(i,j)=layout==101 ? (i-1)*la+j : i+(j-1)*la
                ci(i,j)=layout==101 ? (i-1)*lc+j : i+(j-1)*lc
                a=fill(T(NaN),layout==101 ? max(1,k)*la : max(1,order)*la)
                tau=fill(T(778),k+2);tau[2:k+1].=ext.tau[1:k]
                for i in 1:k,j in 1:order
                    if high||j>order-l;a[ai(i,j)]=j>order-l ? ext.a[i,j] : zero(T);end
                end
                c=fill(T(NaN),layout==101 ? max(1,m)*lc : max(1,n)*lc);mask=falses(length(c))
                for i in 1:m,j in 1:n;c[ci(i,j)]=c0[i,j];mask[ci(i,j)]=true;end
                sa,st,sc=copy(a),copy(tau),copy(c)
                minimum=max(1,side=='L' ? n : m);work=fill(T(780),minimum+2)
                call(lw)=lapacke288_rz_apply(T,J,p,high,layout,side,tr,m,n,k,l,a,la,view(tau,2:k+1),c,lc,view(work,2:minimum+1),lw,name,route)
                if !high
                    @test call(-1)==0
                    @test isequal(a,sa)&&tau==st&&isequal(c,sc)
                    @test real(work[2])>=1
                    if m==0||n==0;@test work[2]==one(T);end
                end
                @test call(minimum)==0
                @test isequal(a,sa)&&tau==st&&isequal(c[.!mask],sc[.!mask])
                @test work[[1,end]]==fill(T(780),2)
                if !high&&(m==0||n==0);@test work[2]==one(T);end
                actual=T[c[ci(i,j)] for i in 1:m,j in 1:n]
                @test actual≈expected atol=tol
            end
        end
    end
end
