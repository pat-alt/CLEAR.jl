using KernelFunctions
using LinearAlgebra

const default_kernel = with_lengthscale(KernelFunctions.GaussianKernel(), 0.5)

"""
    kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix, y::AbstractMatrix)

Compute the sum of kernel matrices between two matrices `x` and `y`. This function sums all kernel evaluations comparing columns in `x` to columns in `y`.
"""
function kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix, y::AbstractMatrix)
    m = size(x, 2)
    n = size(y, 2)
    return sum(kernelmatrix(k, x, y)) / (m * n)
end

"""
    kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix)

Compute the sum of kernel matrices between `x` and itself. This function sums all kernel evaluations comparing columns in `x` to each other and subtracts the trace of the resulting matrix to account for self-evaluations. The result is then divided by `(m^2 - m)` where `m` is the number of columns in `x`. This effectively gives you the mean of all pairwise kernel evaluations excluding self-evaluations.
"""
function kernelsum(k::KernelFunctions.Kernel, x::AbstractMatrix)
    l = size(x, 2)
    return (sum(kernelmatrix(k, x, x)) - tr(kernelmatrix(k, x, x))) / (l^2 - l)
end

"""
    kernelsum(k::KernelFunctions.Kernel, x::AbstractVector)

Compute the sum of kernel matrices between `x` and itself where `x` is a vector. This function returns 0 for vectors since there are no pairs to compute the kernel matrix.
"""
kernelsum(k::KernelFunctions.Kernel, x::AbstractVector) = zero(eltype(x))
