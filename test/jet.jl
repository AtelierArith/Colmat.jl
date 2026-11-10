using Colmat
using JET
using LinearAlgebra
using SparseArrays
using Test

const _jet_modules = (Colmat,)

@testset "JET: representative concrete calls" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        a = FaerMatrix(rand(T, 4, 4))
        b = FaerMatrix(rand(T, 4, 2))
        v = rand(T, 4)
        s = FaerSparseMatrix(sparse(Matrix(a) + 4I))
        @test_call target_modules = _jet_modules a * b
        @test_call target_modules = _jet_modules a + a
        @test_call target_modules = _jet_modules Matrix(a)
        @test_call target_modules = _jet_modules norm(a)
        @test_call target_modules = _jet_modules lu(a)
        @test_call target_modules = _jet_modules qr(a)
        @test_call target_modules = _jet_modules lu(a) \ b
        @test_call target_modules = _jet_modules lu(a) \ v
        @test_call target_modules = _jet_modules eigen(a)
        @test_call target_modules = _jet_modules svd(a)
        @test_call target_modules = _jet_modules schur(a)
        @test_call target_modules = _jet_modules pfaffian(a)
        @test_call target_modules = _jet_modules lu(s) \ b
    end
end

# Factorization `getproperty` methods branch on the kind parameter `K` and fall
# back to `getfield(f, name)`. Package-wide analysis leaves `K` unknown, so JET
# reports the fallback for names that only exist for other kinds. Those reports
# are not reachable at run time and are filtered here.
_kind_field_fallback(r) = r isa JET.BuiltinErrorReport && r.f === getfield

@testset "JET: package-wide undefined names" begin
    result = JET.report_package(
        Colmat; target_modules=_jet_modules, mode=:typo, toplevel_logger=nothing
    )
    reports = filter(!_kind_field_fallback, JET.get_reports(result))
    isempty(reports) || show(stdout, result)
    @test isempty(reports)
end
