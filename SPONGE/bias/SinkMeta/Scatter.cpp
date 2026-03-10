/**
 * @file
 * @author
 * - Zhijun Pan
 */

#include "Scatter.h"

#include <fstream>
#include <istream>

using namespace std;

/**
 * @brief Scatter Class for Enhanced Sampling Methods
 */
template <typename T>
class Scatter : public ScatterBase<T>
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
     * Additionally, the functions SIterator::indices() and
     * SIterator::coordinates() are provided. These functions return the
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
     * Additionaly, the iterator can be shifted by adding or subtracting a
     * vector of ints. The vector needs to have the same dimension as the
     * histogram.
     */
    template <typename R>
    class SIterator
    {
       public:
        using self_type = SIterator;  ///< Type name of the iterato
        using reference = R&;  ///< Either T& or T const& for iterator and
                               ///< const_iterator, respectively.
        // typedef bidirectional_iterator_tag iterator_category; ///<
        // HistIterator is a bidirectional iterator.

        /**
         * @brief  Constructor
         * @param tindices Bin indices specifying the current position of the
         * iterator.
         * @param tgrid Pointer to the grid to iterate over.
         */
        SIterator(const vector<int>& tindices, Scatter<T>* tgrid);

        /**
         * @brief Const constructor
         * @param other SIterator to be copied.
         */
        SIterator(const self_type& other);
        /**
         * @brief  Dereference operator.
         * @return Reference to the value at the current grid position.
         */
        reference operator*(void) { return grid->at(indices); }

        /**
         * @brief  Pre-increment operator.
         * @return Reference to iterator.
         *
         * Increments the bin index of lowest dimension.
         */
        self_type& operator++(void)  //;
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
        self_type& operator--(void)  //;
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
        self_type operator++(int)  //;
        {
            SIterator it(*this);
            ++(*this);
            return it;
        }

        /**
         * @brief Subtraction assignment operator.
         * @param shift Vector to be subtracted from the current grid indices.
         * @return Reference to iterator.
         */
        self_type& operator-=(const vector<int>& shift)  //;
        {
            if (shift.size() != grid->GetDimension())
            {
                printf(
                    "Vector to shift iterator does not match histogram "
                    "dimension.");
            }

            for (size_t i = 0; i < grid->GetDimension(); ++i)
            {
                int index = indices.at(i) + shift.at(i);
                auto numPoints = grid->GetNumPoints();
                int numPoint = numPoints[i];
                if (!grid->GetPeriodic(i))
                {
                    if (index < 0 || index >= numPoint)  // return end();
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
        self_type& operator+=(const vector<int>& shift)  //;
        {
            if (shift.size() != grid->GetDimension())
            {
                printf(
                    "Vector to shift iterator does not match histogram "
                    "dimension.");
            }

            for (size_t i = 0; i < grid->GetDimension(); ++i)
            {
                int index = indices.at(i) + shift.at(i);
                auto numPoints = grid->GetNumPoints();
                int numPoint = numPoints[i];
                if (!grid->GetPeriodic(i))
                {
                    if (index < 0 || index >= numPoint)  //->grid->end();
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

        void SetIndex(const int index, const int i) { indices[i] = index; }
        /**
         * @brief Subtraction iterator
         *
         * @param shift Vector to be subtracted from the current grid indices.
         * @return Copy of iterator after shift.
         */
        const self_type operator-(vector<int> shift)  //;
        {
            return SIterator(*this) -= shift;
        }

        /**
         * @brief Equality operator
         *
         * @param rhs Iterator to which this iterator is compared.
         * @return \c True if both iterators access the same grid point on the
         *         same grid. Else return \c False.
         */
        bool operator==(const self_type& rhs) const  //;
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
        bool operator!=(const self_type& rhs) const
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
        vector<int>& GetIndices(void) { return indices; }
        /**
         * @brief  Access a specific index.
         *
         * @param d Dimension of the index.
         * @return Index of the current grid point in the specified dimension.
         *
         */
        int& GetIndex(size_t d) { return GetIndices()[d]; }
        /**
         * @brief Access coordinates.
         *
         * @return Center point of the current grid point.
         */
        vector<float> coordinates(void) const  //;
        {
            return grid->GetCoordinates(indices);
        }
        /**
         * @brief  Access specific coordinate dimension.
         * @param d Dimension of the coordinate.
         * @return Center of the current grid point in the specified dimension.
         */
        float coordinate(size_t d) const { return coordinates()[d]; }

       private:
        vector<int> indices;  ///< Indices of current grid point.
        Scatter<T>* grid;     ///< Pointer to grid to iterate over.
    };
    using iterator = SIterator<T>;  ///< Custom iterator over a grid.
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
     * @brief Resize the Scatter (based on the read in number of point)
     *
     * @param[in] numPoints The new grid number point vector.
     */
    void Resize(const vector<int>& numPoints);

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
    Scatter(const vector<int>& numPoints, const vector<float>& lower,
            const vector<float>& upper, vector<bool>& isPeriodic);
    Scatter(const vector<int>& numPoints, const vector<float>& period,
            const vector<vector<float>>& coor)
        : ScatterBase<T>(numPoints, period, coor)
    {
        ScatterBase<T>::data_.resize(coor.size());
    }

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
    // static Scatter<T>* BuildScatter(const Json::Value& json);
    ~Scatter(void);

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
    size_t mapTo1d(const vector<int>& indices) const override;
};

using namespace std;
template <typename T>
template <typename R>
Scatter<T>::SIterator<R>::SIterator(const vector<int>& tindices,
                                    Scatter<T>* tgrid)
    : indices(tindices), grid(tgrid)
{
}

template <typename T>
template <typename R>
Scatter<T>::SIterator<R>::SIterator(
    const Scatter<T>::SIterator<R>::self_type& other)
    : indices(other.indices), grid(other.grid)
{
}

template <typename T>
void Scatter<T>::Resize(const vector<int>& numPoints)
{
    /*    size_t data_size = 1;
        for (size_t d = 0; d < ScatterBase<T>::GetDimension(); ++d)
        {
            size_t storage_size = numPoints[d];
            ScatterBase<T>::numPoints_[d] = storage_size;
            data_size *= storage_size;
        }
        ScatterBase<T>::data_.resize(data_size);
        // ScatterBase<T>::syncScatter();
        */
}

template <typename T>
Scatter<T>::Scatter(const vector<int>& numPoints, const vector<float>& lower,
                    const vector<float>& upper, vector<bool>& isPeriodic)
    : ScatterBase<T>(numPoints, lower, upper, isPeriodic)
{
    size_t data_size = 1;
    for (size_t d = 0; d < ScatterBase<T>::GetDimension(); ++d)
    {
        size_t storage_size = ScatterBase<T>::numPoints_[d];
        data_size *= storage_size;
    }
    ScatterBase<T>::data_.resize(data_size);
}
template <typename T>
typename Scatter<T>::template SIterator<T> Scatter<T>::begin(void)
{
    vector<int> indices(ScatterBase<T>::GetDimension(), 0);
    return iterator(indices, this);
}
template <typename T>
typename Scatter<T>::template SIterator<T> Scatter<T>::end(void)
{
    vector<int> indices(ScatterBase<T>::GetDimension());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices.at(i) = ScatterBase<T>::numPoints_[i] - 1;
    }

    iterator it(indices, this);
    return ++it;
}
template <typename T>
size_t Scatter<T>::mapTo1d(const vector<int>& indices) const
{
    // Check if an index is out of bounds
    for (size_t i = 0; i < ScatterBase<T>::GetDimension(); ++i)
    {
        int index = indices.at(i);
        int numpoints = ScatterBase<T>::numPoints_[i];
        if (index < 0 || index >= numpoints)
        {
            printf(
                "Scatter index %zu out of range: index is %d while the number "
                "of points is %d",
                i, index, numpoints);
        }
    }

    size_t idx = 0;
    size_t fac = 1;
    for (size_t i = 0; i < ScatterBase<T>::GetDimension(); ++i)
    {
        idx += indices.at(i) * fac;
        fac *= ScatterBase<T>::numPoints_[i];
    }
    return idx;
}
template <typename T>
Scatter<T>::~Scatter(void)
{
}

template class Scatter<int>;
template class Scatter<int>::SIterator<int>;
template class Scatter<float>;
template class Scatter<float>::SIterator<float>;
template class Scatter<float*>;
template class Scatter<float*>::SIterator<float*>;
