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
    scottgs::FloatMatrix::size_type size,
    scottgs::FloatMatrix::const_iterator1 v1,
    scottgs::FloatMatrix::const_iterator2 v2)
{
    float result = 0.0;
    auto a = v1.cbegin();
    auto b = v2.cbegin();
    for (scottgs::FloatMatrix::size_type i = 0; i < size; i++, a++, b++)
        result += (*a) * (*b);
    return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::operator()(
    const scottgs::FloatMatrix &lhs,
    const scottgs::FloatMatrix &rhs) const
{
    // Verify acceptable dimensions
    if (lhs.size2() != rhs.size1())
        throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

    scottgs::FloatMatrix result(lhs.size1(), rhs.size2());
    auto size = lhs.size2();
    auto row = lhs.cbegin1();
    auto col = rhs.cbegin2();

    for (auto row = lhs.cbegin1(); row != lhs.cend1(); row++)
        for (auto col = rhs.cbegin2(); col != rhs.cend2(); col++)
            result(row.index1(), col.index2()) = dot_product(size, row, col);

    return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix &lhs, const scottgs::FloatMatrix &rhs) const
{
    // Verify acceptable dimensions
    if (lhs.size2() != rhs.size1())
        throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

    return boost::numeric::ublas::prod(lhs, rhs);
}
