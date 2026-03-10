/**
 * @file
 * @author
 * - Zhijun Pan
 */

// clang-format off
#include "GridBase.h"
#include <fstream>
#include <istream>
#include <vector>

using namespace std;

/**
 * @brief Grid Class for Enhanced Sampling Methods
 */
template <typename T>
class Grid : public GridBase<T>
{

public:
    /**
     * @brief  Custom Iterator
     *
     * @details
     * This iterator is designed of travesing through a grid. The starting point
     * is at grid index 0 for each dimension. The last valid grid point has the
     * num_points in each dimension, where num_points is the number of grid
     * points in the respective dimension.
     *
     * The iterator can be used as a standard iterator with operator* accessing
     * the grid point at which the iterator currently is.
     *
     * Additionally, the functions GridIterator::indices() and
     * GridIterator::coordinates() are provided. These functions return the
     * indices of the current grid point and the center of the grid point
     * interval in real space, respectively.
     *
     * The iterator can be moved to an arbitrary position. As indices() returns
     * a reference (and not a const reference), it can be used to move the
     * iterator. For example:
     *
     * \code{.cpp}
     * HistIterator it = hist->begin();
     * it.indices() = {1,1,1};
     * \endcode
     *
     * moves the iterator to the grid point [1, 1, 1].
     *
     * The iterator can be traversed in a standard fashion with the increment
     * and decrement operators operator++ and operator--. When the increment
     * operator is invoked, the bin index for the lowest dimension is increased
     * by 1. If it moves beyond the valid range in this dimension, the index is
     * reset to 0 and the index of the next higher dimension is increased by 1.
     * The decrement operator traveses the grid in the same fashion but opposite
     * direction.
     *
     * Additionaly, the iterator can be shifted by adding or subtracting a vector
     * of ints. The vector needs to have the same dimension as the histogram.
     */
    template <typename R>
    class GridIterator
    {
    public:
        using self_type = GridIterator; ///< Type name of the iterato
        using reference = R &;          ///< Either T& or T const& for iterator and const_iterator, respectively.
        // typedef bidirectional_iterator_tag iterator_category; ///< HistIterator is a bidirectional iterator.

        /**
         * @brief  Constructor
         * @param tindices Bin indices specifying the current position of the  iterator.
         * @param tgrid Pointer to the grid to iterate over.
         */
        GridIterator(const vector<int> &tindices, Grid<T> *tgrid);

        /**
         * @brief Const constructor
         * @param other GridIterator to be copied.
         */
        GridIterator(const self_type &other);
        /**
         * @brief  Dereference operator.
         * @return Reference to the value at the current grid position.
         */
        reference operator*(void)
        {
            return grid->at(indices);
        }

        /**
         * @brief  Pre-increment operator.
         * @return Reference to iterator.
         *
         * Increments the bin index of lowest dimension. If an index moves
         * beyond the maximum value (num_points-1), it is reset to 0 and the
         * index of the next higher dimension is increased by 1.
         */
        self_type &operator++(void) //;
        {
            indices.at(0) += 1;
            for (size_t i = 0; i < grid->GetDimension() - 1; ++i)
            {
                if (indices.at(i) >= grid->numPoints_[i])
                {
                    indices.at(i) = 0;
                    indices.at(i + 1) += 1;
                }
            }

            return *this;
        }
        self_type &operator--(void) //;
        {
            indices.at(0) -= 1;
            for (size_t i = 0; i < grid->GetDimension() - 1; ++i)
            {
                if (indices.at(i) < 0)
                {
                    indices.at(i) = grid->numPoints_[i] - 1;
                    indices.at(i + 1) -= 1;
                }
            }

            return *this;
        }
        /**
         * @brief /! Post-increment operator.
         * @return Copy of iterator before incrementing.
         */
        self_type operator++(int) //;
        {
            GridIterator it(*this);
            ++(*this);
            return it;
        }

        /**
         * @brief Subtraction assignment operator.
         * @param shift Vector to be subtracted from the current grid indices.
         * @return Reference to iterator.
         */
        self_type &operator-=(const vector<int> &shift) //;
        {
            if (shift.size() != grid->GetDimension())
            {
                printf("Vector to shift iterator does not match histogram dimension.");
            }

            for (size_t i = 0; i < grid->GetDimension(); ++i)
            {
                int index = indices.at(i) + shift.at(i);
                auto numPoints = grid->GetNumPoints();
                int numPoint = numPoints[i];
                if (!grid->GetPeriodic(i))
                {
                    if (index < 0 || index >= numPoint) // return end();
                    {
                        for (size_t i = 0; i < indices.size(); ++i)
                        {
                            indices.at(i) = numPoint - 1;
                        }

                        return ++*this;
                    }
                }
                else
                {
                    while (index < 0)
                    {
                        index += numPoint;
                    }
                    while (index >= numPoint)
                    {
                        index -= numPoint;
                    }
                }
                indices.at(i) = index;
            }

            return *this;
        }

        /**
         * @brief Sum up assignment operator.
         * @param shift Vector to be summed from the current grid indices.
         * @return Reference to iterator.
         */
        self_type &operator+=(const vector<int> &shift) //;
        {
            if (shift.size() != grid->GetDimension())
            {
                printf("Vector to shift iterator does not match histogram dimension.");
            }

            for (size_t i = 0; i < grid->GetDimension(); ++i)
            {
                int index = indices.at(i) + shift.at(i);
                auto numPoints = grid->GetNumPoints();
                int numPoint = numPoints[i];
                if (!grid->GetPeriodic(i))
                {
                    if (index < 0 || index >= numPoint) //->grid->end();
                    {
                        for (size_t i = 0; i < indices.size(); ++i)
                        {
                            indices.at(i) = numPoint - 1;
                        }

                        return ++*this;
                    }
                }
                else
                {
                    while (index < 0)
                    {
                        index += numPoint;
                    }
                    while (index >= numPoint)
                    {
                        index -= numPoint;
                    }
                }
                indices.at(i) = index;
            }
            // indices = grid->wrapIndices(indices);

            return *this;
        }

        void SetIndex(const int index, const int i)
        {
            indices[i] = index;
        }
        /**
         * @brief Subtraction iterator
         *
         * @param shift Vector to be subtracted from the current grid indices.
         * @return Copy of iterator after shift.
         */
        const self_type operator-(vector<int> shift) //;
        {
            return GridIterator(*this) -= shift;
        }

        /**
         * @brief Equality operator
         *
         * @param rhs Iterator to which this iterator is compared.
         * @return \c True if both iterators access the same grid point on the
         *         same grid. Else return \c False.
         */
        bool operator==(const self_type &rhs) const //;
        {
            return indices == rhs.indices && grid == rhs.grid;
        }

        /**
         * @brief Non-equality operator.
         *
         * @param rhs Iterator to which this iterator is compared.
         * @return \c False if both iterators access the same grid point on the
         *         same grid. Else return \c True.
         */
        bool operator!=(const self_type &rhs) const
        {
            return !((*this) == rhs);
        }

        /**
         * @brief  Access indices.
         *
         * @return Indices of current grid point.
         *
         * \note This function returns a reference and can be used to move the
         *       current grid point.
         */
        vector<int> &GetIndices(void)
        {
            return indices;
        }
        /**
         * @brief  Access a specific index.
         *
         * @param d Dimension of the index.
         * @return Index of the current grid point in the specified dimension.
         *
         */
        int &GetIndex(size_t d)
        {
            return GetIndices()[d];
        }
        /**
         * @brief Access coordinates.
         *
         * @return Center point of the current grid point.
         */
        vector<float> coordinates(void) const //;
        {
            return grid->GetCoordinates(indices);
        }
        /**
         * @brief  Access specific coordinate dimension.
         * @param d Dimension of the coordinate.
         * @return Center of the current grid point in the specified dimension.
         */
        float coordinate(size_t d) const
        {
            return coordinates()[d];
        }

    private:
        vector<int> indices; ///< Indices of current grid point.
        Grid<T> *grid;       ///< Pointer to grid to iterate over.
    };
    using iterator = GridIterator<T>; ///< Custom iterator over a grid.
    /**
     * @brief  Return iterator at first grid point.
     *
     * @return Iterator at first grid point.
     *
     * The first grid point is defined as the grid point with index 0 in all
     * dimensions.
     */
    iterator begin(void);
    /**
     * @brief Return iterator after last valid grid point.
     * @return Iterator after last valid grid point.
     *
     * The last valid grid point has index num_points - 1 in all dimensions.
     */
    iterator end(void);
    /**
     * @brief Resize the Grid (based on the read in number of point)
     *
     * @param[in] numPoints The new grid number point vector.
     */
    void Resize(const vector<int> &numPoints);

    /**
     * @brief Constructor
     *
     * @param numPoints Number of grid points in each dimension.
     * @param lower Lower edges of the grid.
     * @param upper Upper edges of the grid.
     * @param isPeriodic Bools specifying the periodicity in the respective
     *                   dimension.
     *
     * The dimension of the grid is determined by the size of the parameter
     * vectors.
     */
    Grid(const vector<int> &numPoints, const vector<float> &lower, const vector<float> &upper,
         vector<bool> &isPeriodic);

    /**
     * @brief  Set up the grid
     *
     * @param json JSON value containing all input information.
     * @return Pointer to the newly built grid.
     *
     * This function builds a grid from a JSON node. It will return a nullptr
     * if an unknown error occured, but generally, it will throw a
     * BuildException of failure.
     */
    // static Grid<T>* BuildGrid(const Json::Value& json);
    ~Grid(void);

    /**
     * @brief Compute Del(nabla) matrix for linear system.
     * @details Forward diff(1st order) is enough: \f$ \mathbf{\Delta} = \mathbf{T(r)} - \mathbf{I} \f$
     *
     * For function f(x),
     *       The shift operator of the r defined as
     *       \f[
     *       \mathbf{T(r)} f(x)  \equiv f(x+r)
     *       \f]
     *
     * https://www.intechopen.com/chapters/59479
     * @return MatrixXd A as gradient operater.
     */

private:
    /**
     * @brief Map d-dimensional indices to 1-d data vector
     *
     * @param indices Vector specifying the grid point.
     * @return Index of 1d data vector.
     *
     * Map a set of indices to the index of the 1d data vector. Keep in mind,
     * that the data includes underflow (index -1) and overflow (index
     * numPoints) bins in periodic dimension.
     */
    size_t mapTo1d(const vector<int> &indices) const override;
};

using namespace std;
template <typename T>
template <typename R>
Grid<T>::GridIterator<R>::GridIterator(const vector<int> &tindices, Grid<T> *tgrid) : indices(tindices),
                                                                                      grid(tgrid)
{
}

template <typename T>
template <typename R>
Grid<T>::GridIterator<R>::GridIterator(const Grid<T>::GridIterator<R>::self_type &other)
    : indices(other.indices),
      grid(other.grid)
{
}
/*
template <typename T>
template <typename R>
typename Grid<T>::GridIterator<R>::R& Grid<T>::GridIterator<R>::operator*(void)
{
    return grid->at(indices);
}
template <typename T>
template <typename R>
typename Grid<T>::GridIterator<R>::self_type& Grid<T>::GridIterator<R>::operator++(void)
template <typename T>
template <typename R>
typename Grid<T>::GridIterator<R>::self_type Grid<T>::GridIterator<R>::operator++(int)

template <typename T>
template <typename R>
typename Grid<T>::GridIterator<R>::self_type& Grid<T>::GridIterator<R>::operator+=(const vector<int>& shift)
*/

template <typename T>
void Grid<T>::Resize(const vector<int> &numPoints)
{
    size_t data_size = 1;
    for (size_t d = 0; d < GridBase<T>::GetDimension(); ++d)
    {
        size_t storage_size = numPoints[d];
        GridBase<T>::numPoints_[d] = storage_size;
        data_size *= storage_size;
    }
    GridBase<T>::data_.resize(data_size);
    // GridBase<T>::syncGrid();
}

template <typename T>
Grid<T>::Grid(const vector<int> &numPoints, const vector<float> &lower, const vector<float> &upper,
              vector<bool> &isPeriodic)
    : GridBase<T>(numPoints, lower, upper, isPeriodic)
{
    size_t data_size = 1;
    for (size_t d = 0; d < GridBase<T>::GetDimension(); ++d)
    {
        size_t storage_size = GridBase<T>::numPoints_[d];
        data_size *= storage_size;
    }
    GridBase<T>::data_.resize(data_size);
}
template <typename T>
typename Grid<T>::template GridIterator<T> Grid<T>::begin(void)
{
    vector<int> indices(GridBase<T>::GetDimension(), 0);
    return iterator(indices, this);
}
template <typename T>
typename Grid<T>::template GridIterator<T> Grid<T>::end(void)
{
    vector<int> indices(GridBase<T>::GetDimension());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices.at(i) = GridBase<T>::numPoints_[i] - 1;
    }

    iterator it(indices, this);
    return ++it;
}
template <typename T>
size_t Grid<T>::mapTo1d(const vector<int> &indices) const
{
    // Check if an index is out of bounds
    for (size_t i = 0; i < GridBase<T>::GetDimension(); ++i)
    {
        int index = indices.at(i);
        int numpoints = GridBase<T>::numPoints_[i];
        if (index < 0 || index >= numpoints)
        {
            printf("Grid index %zu out of range: index is %d while the number of points is %d", i, index,
                   numpoints);
        }
    }

    size_t idx = 0;
    size_t fac = 1;
    for (size_t i = 0; i < GridBase<T>::GetDimension(); ++i)
    {
        idx += indices.at(i) * fac;
        fac *= GridBase<T>::numPoints_[i];
    }
    return idx;
}
template <typename T>
Grid<T>::~Grid(void) {}

template class Grid<int>;
template class Grid<int>::GridIterator<int>;
template class Grid<float>;
template class Grid<float>::GridIterator<float>;
template class Grid<float *>;
template class Grid<float *>::GridIterator<float *>;
