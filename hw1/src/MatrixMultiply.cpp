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
    scottgs::FloatMatrix::value_type rhs[],
    scottgs::FloatMatrix::size_type size,
    scottgs::FloatMatrix::size_type row)
{
    float result = 0.0;
    for (scottgs::FloatMatrix::size_type i = 0; i < size; i++)
        result += lhs[row * size + i] * rhs[i];
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

    auto lhs_n_rows = lhs_m.size1(),
         rhs_n_rows = rhs_m.size1(), rhs_n_cols = rhs_m.size2();

    const scottgs::FloatMatrix::array_type &lhs = lhs_m.data();

    const scottgs::FloatMatrix::array_type &rhs = rhs_m.data();
    scottgs::FloatMatrix::value_type *col_trans =
        new scottgs::FloatMatrix::value_type[rhs_n_rows];

    for (scottgs::FloatMatrix::size_type b = 0; b < rhs_n_cols; b++)
    {
        // Transpose column about to be used
        for (scottgs::FloatMatrix::size_type i = 0; i < rhs_n_rows; i++)
            col_trans[i] = rhs[i * rhs_n_cols + b];
        // Do calculations
        for (scottgs::FloatMatrix::size_type a = 0; a < lhs_n_rows; a++)
            result[a * rhs_n_cols + b] = dot_product(lhs, col_trans, rhs_n_rows, a);
    }

    delete[] col_trans;
    return result_m;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix &lhs, const scottgs::FloatMatrix &rhs) const
{
    // Verify acceptable dimensions
    if (lhs.size2() != rhs.size1())
        throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

    return boost::numeric::ublas::prod(lhs, rhs);
}
