/**
 * This file is from SSAGES
 */
#ifndef __SCATTER_CUH__
#define __SCATTER_CUH__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

/**
 * @brief Base class for Grids
 * \tparam Type of data stored on the grid
 *
 */
template <typename T>
class ScatterBase
{
   public:
    /**
     * @brief  Get the dimension.
     *
     * \return Dimensionality of the grid.
     */
    size_t GetDimension() const { return dimension_; }

    /**
     * @brief Get the number of points for all dimensions.
     *
     * \return Vector of ints containing the number of grid points for each
     *         dimension.
     */
    const vector<int> GetNumPoints() const { return numPoints_; }

    /**
     * @brief /! Get the number of points for a specific dimension.
     *
     * @param dim Index of the dimension.
     * \return Number of grid points in the requested dimension.
     *
     * \note The first dimension uses the index 0.
     */
    int GetNumPoints(size_t dim) const
    {
        if (dim >= GetDimension())
        {
            printf(
                "Warning! Grid size requested for a dimension larger than the "
                "grid dimensionality!\n");
            return 0;
        }
        return numPoints_.at(dim);
    }

    /**
     * @brief  Return the periodicity of the Grid.
     *
     * \return Vector of bools. The values are \c True (\c False ) if the grid
     *         is periodic (non-periodic) in the given dimension.
     */
    const vector<bool>& GetPeriodic() const { return isPeriodic_; }

    /**
     * @brief  Get the periodicity in a specific dimension.
     *
     * @param dim Index of the dimension.
     * \return \c True (\c False ) if the grid is periodic (non-periodic) in
     *         the specified dimension.
     *
     * \note The dimensions are indexed starting with 0.
     */
    bool GetPeriodic(size_t dim) const
    {
        if (dim >= GetDimension())
        {
            printf(
                "Warning! Periodicity requested for a dimension larger than "
                "the grid dimensionality!\n");
            return false;
        }
        return GetPeriodic().at(dim);
    }

    /**
     * @brief  Get the size of the internal storage vector
     *
     * \return Size of the internal storage vector.
     *
     * This function returns the size of the internal storage vector. This is
     * also the total number of grid points including the over/underflow bins
     * in case of a histogram.
     */
    size_t size() const { return data_.size(); }

    /**
     * @brief  Get pointer to the internal data storage vector
     * \return Pointer to data in the internal storage vector.
     *
     * It is discouraged to directly access the internal data storage. It might,
     * however be necessary. For example when communicating the data over MPI.
     */
    T* data() { return data_.data(); }

    /**
     * @brief Get pointer to const of the internal data storage vector
     *
     * \return Const pointer to data in the internal storage vector.
     *
     * It is discouraged to directly access the internal data storage. It might,
     * however be necessary. For example when communicating data over MPI.
     */
    T const* data() const { return data_.data(); }

    int GetIndex(const vector<float>& x) const
    {
        // Check that input vector has the correct dimensionality
        if (x.size() != dimension_)
        {
            printf(
                "Specified point: %ld has a different dimensionality than the "
                "grid: %ld.",
                x.size(), dimension_);
        }
        // Metric distance2 as sum_i(x_i-coor_i)^2
        vector<float> x_distance(coordinate.size());
        // Lambda function
        auto ff = [=](vector<float> v) { // return (v[0] - x[0]) * (v[0] - x[0]) + (v[1] - x[1]) * (v[1] - x[1]);
            float sum_square = 0.;
            for (size_t i = 0; i < dimension_; ++i)
            {
                float distance = v[i] - x[i];
                if (GetPeriodic(i))
                {
                    distance -= round(distance / period[i]) * period[i];
                }
                sum_square += distance * distance;
            }
            return sum_square;
        };
        transform(coordinate.begin(), coordinate.end(), x_distance.begin(), ff);
        std::vector<float>::iterator it =
            min_element(x_distance.begin(), x_distance.end());

        return distance(x_distance.begin(), it);
    }

    /**
     * @brief  Return the Grid indices for a given point.
     *
     * @param x Point in space.
     * \return Indices of the grid point to which the point in space pertains.
     *
     */
    vector<int> GetIndices(const vector<float>& x) const
    {
        // Check that input vector has the correct dimensionality
        if (x.size() != dimension_)
        {
            printf(
                "Specified point: %ld has a different dimensionality than the "
                "grid: %ld.",
                x.size(), dimension_);
        }
        // Metric distance2 as sum_i(x_i-coor_i)^2
        vector<float> x_distance(coordinate.size());
        // Lambda function
        auto ff = [=](vector<float> v) { // return (v[0] - x[0]) * (v[0] - x[0]) + (v[1] - x[1]) * (v[1] - x[1]);
            float sum_square = 0.;
            for (size_t i = 0; i < dimension_; ++i)
            {
                const float &distance = v[i] - x[i];
                sum_square += distance * distance;
            }
            return sum_square;
        };
        transform(coordinate.begin(), coordinate.end(), x_distance.begin(), ff);
        std::vector<float>::iterator it =
            min_element(x_distance.begin(), x_distance.end());
        int i = distance(x_distance.begin(), it);

        return wrapIndices(i);
    }

    /**
     * @brief /! Return the Grid index for a one-dimensional grid.
     * @param x Point in space.
     * \return Grid index to which the point pertains.
     *
     * Return the Grid index pertaining to the given point in space. This
     * function is for convenience when accessing 1d-Grids. For
     * higher-dimensional grids, x needs to be a vector of floats.
    int GetIndex(float x) const
    {
        if (dimension_ != 1)
        {
            printf("1d Grid index can only be accessed for 1d-Grids can be
    accessed with a.");
        }
        return GetIndices({x}).at(0);
    }
     */

    /**
     * @brief /! Return coordinates of the grid center points
     * @param indices Grid indices specifying a grid point.
     * \return Vector of float specifying the position of the grid point.
     *
     * The grid is a discretization of real or cv space. Thus, each grid point
     * is associated with an interval of the underlying space. This function
     * returns the center point of this interval.
     */
    vector<float> GetCoordinates(const vector<int>& indices)
    {
        if (indices.size() != dimension_)
        {
            printf(
                "Grid indices specified for GetCoordinates() do not have the "
                "same dimensionality as the grid.");
        }

        return coordinate.at(mapTo1d(indices));
    }

    /**
     * @brief  Return center point of 1d-grid
     * @param index Index of the 1d grid.
     * \return Coordinate in real/CV space.
     *
     * \note This function is only available for 1d grids.
    float GetCoordinate(int index)
    {
        return GetCoordinates({index}).at(0);
    }
     */
    vector<float> GetCoordinate(int index)
    {
        return coordinate.at(index);  // GetCoordinates({index}).at(0);
    }
    vector<int> GetNeighbor(const vector<float>& values,
                            const float* cutoff) const
    {
        vector<int> neighber_index;
        for (int index = 0; index < coordinate.size(); ++index)
        {
            bool isneighbor = true;
            for (int i = 0; i < dimension_; ++i)
            {
                float distance = coordinate[index][i] - values[i];
                if (isPeriodic_[i])
                {
                    distance -= roundf(distance / period[i]) * period[i];
                }
                if (fabs(distance) > cutoff[i])
                {
                    isneighbor = false;  // not nieghbor!
                    break;
                }
            }
            if (isneighbor)
            {
                neighber_index.push_back(index);
            }
        }
        return neighber_index;
    }
    /**
     * @brief Access Grid element read-only
     * @param indices Vector of integers specifying the grid point.
     * \return const reference of the value stored at the given grid point.
     *
     * In non-periodic dimensions, the index needs to be in the interval
     * [-1, numPoints]. Grid::at(-1) accessed the underflow bin,
     * Grid::at(numPoints) accesses the overflow bin.
     *
     * In periodic dimensions, the index may take any integer value and will be
     * mapped back to the interval [0, numPoints-1]. Thus, Grid::at(-1) will
     * access the same value as Grid::at(numPoints-1).
     */
    const T& at(const vector<int>& indices) const
    {
        // Check that indices are in bound.
        if (indices.size() != GetDimension())
        {
            // printf("Dimension of indices does not match dimension of the
            // grid."); int i = indices[0]; int size = data_.size();
            // printf("indices of dimension 0 is %d while the data have
            // %d.\n",i,size);
            return data_[indices[0]];
        }

        return data_.at(mapTo1d(indices));
    }

    /**
     * @brief  Access Grid element read/write
     *
     * @param indices Vector of integers specifying the grid point.
     * \return Reference to value at the specified grid point.
     */
    T& at(const vector<int>& indices)
    {
        return const_cast<T&>(
            static_cast<const ScatterBase<T>*>(this)->at(indices));
    }

    /**
     * @brief  Const access of Grid element via initializer list
     * \tparam R Datatype in the initializer list
     * @param x initializer list
     * \return Const reference to value at the specified point.
     *
     * This function avoids abiguity if at() is called with a brace-enclosed
     * initializer list. The template parameter makes sure that this function
     * can be called with either ints, specifying a grid point, or floats,
     * specifying coordinates in space, inside the initializer list.
     */
    template <typename R>
    const T& at(initializer_list<R>&& x) const
    {
        return at(static_cast<vector<R>>(x));
    }

    /**
     * @brief /! Access Grid element via initializer list
     * \tparam R Datatype in the initializer list
     * @param x initializer list
     * \return Reference to value at the specified point.
     *
     * This function avoids abiguity if at() is called with a brace-enclosed
     * initializer list. The template parameter makes sure that this function
     * can be called with either ints, specifying a grid point, or floats,
     * specifying coordinates in space, inside the initializer list.
     */
    template <typename R>
    T& at(initializer_list<R>&& x)
    {
        return at(static_cast<vector<R>>(x));
    }

    /**
     * @brief  Access 1d Grid by index, read-only
     * @param index Index specifying the grid point.
     * \return Const reference of value at the given index.
     *
     * \note This function can only be used for 1d-Grids.
     */
    const T& at(int index) const { return at({index}); }

    /**
     * @brief  Access 1d Grid by index, read-write
     * @param index Index specifying the grid point.
     * \return Reference of value at the given grid point.
     *
     * \note This function can only be used for 1d-Grids.
     */
    T& at(int index)
    {
        return const_cast<T&>(
            static_cast<const ScatterBase<T>*>(this)->at(index));
    }

    /**
     * @brief  Access Grid element pertaining to a specific point -- read-only
     * @param x Vector of floats specifying a point.
     * \return Const reference of the value at the given coordinates.
     *
     * This function is provided for convenience. It is identical to
     */
    const T& at(const vector<float>& x) const { return at(GetIndex(x)); }

    const T& at(const vector<float>& x, vector<float>& coor) const
    {
        int index = GetIndex(x);
        coor = coordinate[index];
        return at(index);
    }
    /**
     * @brief  Access Grid element pertaining to a specific point -- read/write
     * @param x Vector of floats specifying a point.
     * \return Reference to the value at the given coordinates.
     *
     * This function is provided for convenience. It is identical to
     */
    T& at(const vector<float>& x) { return at(GetIndex(x)); }

    T& at(const vector<float>& x, vector<float>& coor)
    {
        int index = GetIndex(x);
        coor = coordinate[index];
        return at(index);
    }
    /**
     * @brief  Access 1d-Grid by point - read-only
     * @param x Access grid point pertaining to this value.
     * \return Const reference to the value pertaining to the specified
     *         coordinate.
     *
     * \note This function can only be used for 1d-Grids.
     */
    const T& at(float x) const
    {
        if (dimension_ != 1)
        {
            printf(
                "Only 1d-Grids can be accessed with a single float as the "
                "specified point.");
        }
        return at({x});
    }

    /**
     * @brief  Access 1d-Grid by point - read-write
     * @param x Access grid point pertaining to this value.
     * \return Reference to the value pertaining to the specified coordinate.
     *
     * \note This function can only be used for 1d Grids.
     */
    T& at(float x)
    {
        return const_cast<T&>(static_cast<const ScatterBase<T>*>(this)->at(x));
    }

    //! Access Grid element per [] read-only
    /*!
     * @param indices Vector of integers specifying the grid point.
     * \return Const reference to value to the given grid point.
     */
    const T& operator[](const vector<int>& indices) const
    {
        return at(indices);
    }

    //! Access Grid element per [] read-write
    /*!
     * @param indices Vector of integers specifying the grid point.
     * \return Reference to value at the specified grid point.
     */
    T& operator[](const vector<int>& indices) { return at(indices); }

    //! Const access of Grid element via initializer list
    /*!
     * \tparam R Datatype in the initializer list
     * @param x initializer list
     * \return Const reference to the value at the specified point.
     *
     * This function avoids abiguity if operator[] is called with a
     * brace-enclosed initializer list.
     *
     * Example: grid[{0,1}] or grid[{-1.23, 4.2, 0.0}]
     */
    template <typename R>
    const T& operator[](initializer_list<R>&& x) const
    {
        return at(static_cast<vector<R>>(x));
    }

    //! Access Grid element via initializer list
    /*!
     * \tparam R Datatype in the initializer list
     * @param x initializer list
     * \return Reference to the value at the specified point.
     *
     * This function avoids abiguity if operator[] is called with a
     * brace-enclosed initializer list.
     *
     * Example: grid[{0,1}]
     */
    template <typename R>
    T& operator[](initializer_list<R>&& x)
    {
        return at(static_cast<vector<R>>(x));
    }

    //! Access 1d-Grid per [] operator, read-only
    /*!
     * @param index Index of the grid point.
     * \return Const reference to the value at the given grid point.
     *
     * \note This function can only be used for 1d grids.
     */
    const T& operator[](int index) const { return at(index); }

    //! Access 1d-Grid per [] operator, read-write
    /*!
     * @param index Index of the grid point.
     * \return Reference to the value at the given grid point.
     *
     * \note This function can only be used for 1d grids.
     */
    T& operator[](int index) { return at(index); }

    //! Access Grid element pertaining to a specific point per [] read-only
    /*!
     * @param x Vector of floats specifying the point in space.
     * \return Const reference to the value pertaining to the given coordinates.
     */
    const T& operator[](const vector<float>& x) const { return at(x); }

    //! Access Grid element pertaining to a specific point per [] read-write
    /*!
     * @param x Vector of floats specifying the point in space.
     * \return Reference to the value pertaining to the given coordinates.
     */
    T& operator[](const vector<float>& x) { return at(x); }

    //! Access 1d-Grid via specific point, read-only
    /*!
     * @param x Point specifying the desired Grid point.
     * \return Const reference to the value at the specified coordinate.
     *
     * \note This function can only be used for 1d grids.
     */
    const T& operator[](float x) const { return at(x); }

    //! Access 1d-Grid via specific point, read-write
    /*!
     * @param x Point specifying the desired Grid point.
     * \return Reference to value at the specified coordinate.
     *
     * \note This function can only be used for 1d grids.
     */
    T& operator[](float x) { return at(x); }

    //! Destructor
    virtual ~ScatterBase() {}

    vector<int> wrapIndices(const int& igrid) const
    {
        vector<int> indices;
        int index = igrid;
        for (int i = dimension_ - 1; i > -1; --i)
        {
            int remain = floor(index / suffix[i]);
            indices.insert(indices.begin(), remain);
            index -= suffix[i] * remain;
        }
        return indices;
    }
    //! Wrap the index around periodic boundaries
    /*!
     * @param indices Current indices of grid to wrap
     *
     * \return Resulting indices after wrapping
     */
    vector<int> wrapIndices(const vector<int>& indices) const
    {
        vector<int> newIndices(indices);
        for (size_t i = 0; i < dimension_; ++i)
        {
            if (!GetPeriodic(i))
            {
                continue;
            }

            int index = indices.at(i);
            while (index < 0)
            {
                index += numPoints_[i];
            }
            while (index >= numPoints_[i])
            {
                index -= numPoints_[i];
            }
            newIndices.at(i) = index;
        }

        return newIndices;
    }
    vector<T> data_;  ///< Internal storage of the data
    vector<vector<float>>
        coordinate;  ///< Save coordinate, dont compute every time
   protected:
    //! This function needs to be implemented by child classes.
    /*!
     * @param indices The indices specifying the grid point.
     * \return Index of the grid point in the 1d storage vector.
     *
     * mapTo1d maps the indices onto a single index to access the data stored
     * in the data_ vector. This mapping will be different for different grid
     * implementations.
     */
    virtual size_t mapTo1d(const vector<int>& indices) const = 0;
    /**
     * @brief  Constructor
     * @param numPoints Number of grid points in each dimension.
     * @param lower Lower edges of the grid.
     * @param upper Upper edges of the grid.
     * @param isPeriodic Bools specifying the periodicity in the respective
     *                   dimension.
     *
     * The constructor is protected by design. This makes sure that only child
     * classes of ScatterBase are constructed.
     */
    ScatterBase(const vector<int>& numPoints, const vector<float>& lower,
                const vector<float>& upper, const vector<bool>& isPeriodic)
        : dimension_(numPoints.size()),
          numPoints_(numPoints),
          isPeriodic_(isPeriodic)
    {
        if (isPeriodic_.size() == 0)  // Default: Non-periodic in all dimensions
        {
            isPeriodic_.resize(dimension_, false);
        }
        vector<float> spacing;
        int total_size = 1;
        // for (int i = dimension_-1; i >-1 ; --i)
        for (int i = 0; i < dimension_; ++i)
        {
            spacing.push_back((upper[i] - lower[i]) / numPoints_[i]);
            suffix.push_back(total_size);
            total_size *= numPoints_[i];
        }
        coordinate.clear();
        for (int igrid = 0; igrid < total_size; ++igrid)  // reverse of mapTo1d
        {
            int index = igrid;
            vector<float> axis;
            for (int i = dimension_ - 1; i > -1; --i)
            {
                int remain = floor((index) / suffix[i]);
                axis.insert(axis.begin(),
                            lower[i] + (remain + 0.5) * spacing[i]);
                index -= suffix[i] * remain;
                // printf(" %d index is %d,",i,remain);
            }
            coordinate.push_back(axis);
        }
    }
    ScatterBase(const vector<int>& numPoints, const vector<float>& tperiod,
                const vector<vector<float>>& coor)
        : dimension_(numPoints.size()),
          numPoints_(numPoints),
          period(tperiod),
          coordinate(coor)
    {
        int total_size = 1;
        for (int i = 0; i < dimension_; ++i)
        {
            total_size *= numPoints_[i];
            suffix.push_back(
                total_size);  ///< Suffix of points in each dimension.
            isPeriodic_.push_back(period[i] > 0 ? true : false);
        }
        if (coor.size() == 0)
        {
            printf("No coordinate!\n");
            return;
        }
        else if (coor.size() > total_size)
        {
            printf("Too many coordinate!\n");
            return;
        }
    }

    size_t dimension_;         ///< Dimension of the grid
    vector<int> numPoints_;    ///< Number of points in each dimension.
    vector<bool> isPeriodic_;  ///< Periodicity of the Grid.
    vector<float> period;      ///< Periodic shift of the Grid.
    vector<int> suffix;        ///< Suffix of points in each dimension.
};

#endif  // __SCATTER_CUH__
