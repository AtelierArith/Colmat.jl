using Colmat
using ParallelTestRunner

ParallelTestRunner.runtests(
    Colmat,
    ARGS;
    testsuite = ParallelTestRunner.find_tests(@__DIR__),
)
