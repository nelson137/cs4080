#include "MatrixMultiply.hpp"

#include <exception>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>

scottgs::MatrixMultiply::MatrixMultiply()
{
    ;
}

scottgs::MatrixMultiply::~MatrixMultiply()
{
    ;
}

static inline float dot_product(
    scottgs::FloatMatrix::array_type lhs,
    scottgs::FloatMatrix::size_type lhs_width,
    scottgs::FloatMatrix::array_type rhs,
    scottgs::FloatMatrix::size_type rhs_width,
    scottgs::FloatMatrix::size_type r,
    scottgs::FloatMatrix::size_type c)
{
    float result = 0.0;
    for (scottgs::FloatMatrix::size_type i = 0; i < lhs_width; i++)
        result += lhs[r * lhs_width + i] * rhs[i * rhs_width + c];
    return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::operator()(
    const scottgs::FloatMatrix &lhs_m,
    const scottgs::FloatMatrix &rhs_m) const
{
    // Verify acceptable dimensions
    if (lhs_m.size2() != rhs_m.size1())
        throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

    scottgs::FloatMatrix result_m(lhs_m.size1(), rhs_m.size2());
    scottgs::FloatMatrix::array_type &result = result_m.data();
    const scottgs::FloatMatrix::array_type &lhs = lhs_m.data();
    const scottgs::FloatMatrix::array_type &rhs = rhs_m.data();
    auto lhs_width = lhs_m.size2(), rhs_width = rhs_m.size2();

    for (scottgs::FloatMatrix::size_type r = 0; r < lhs_m.size1(); r++)
        for (scottgs::FloatMatrix::size_type c = 0; c < rhs_m.size2(); c++)
            result[r * rhs_width + c] =
                dot_product(lhs, lhs_width, rhs, rhs_width, r, c);

    return result_m;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix &lhs, const scottgs::FloatMatrix &rhs) const
{
    // Verify acceptable dimensions
    if (lhs.size2() != rhs.size1())
        throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

    return boost::numeric::ublas::prod(lhs, rhs);
}
