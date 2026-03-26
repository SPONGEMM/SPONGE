/**
 * This file is from SSAGES
 */
#ifndef __GRIDBASE_H__
#define __GRIDBASE_H__

#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

/**
 * @brief Base class for Grids
 * \tparam Type of data stored on the grid
 *
 * Base class for all grids. Currently, these are 'Grid' and 'Histogram'.
 *
 */
template <typename T>
class GridBase
{
   public:
    //! Sync the grid.
    void syncGrid()
    {
        // Convenience
        size_t dim = this->GetDimension();

        // Preallocate
        vector<int> navigate(dim);

        // Loop over all surfaces. Number of surfaces = dim.
        for (size_t i = 0; i < dim; i++)
        {
            /* Check if periodic in this dimension.
            if (isPeriodic_[i])
            {

                // Calculate surface size. This is equal to number of points in
            each other dimension multiplied. size_t surfsize = 1; for (size_t j
            = 0; j < dim; j++)
                {
                    if (i != j)
                        surfsize *= numPoints_[j];
                }

                // Loop over all points of this surface on the 0 side and copy
            to end. for (size_t j = 0; j < surfsize; j++)
                {
                    int runningcount = j;
                    for (size_t k = 0; k < dim; k++)
                    {
                        if (k == i)
                            navigate[k] = 0;
                        else
                        {
                            navigate[k]  = runningcount % numPoints_[k];
                            runningcount = runningcount / numPoints_[k];
                        }
                    }
                    auto temp          = this->at(navigate);
                    navigate[i]        = numPoints_[i] - 1;
                    this->at(navigate) = temp;
                }
            }
            */
        }
    }

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
     * @brief Return the lower edges of the Grid.
     *
     * \return Vector containing the lower edges of the grid.
     */
    const vector<float> GetLower() const
    {
        vector<float> lower(dimension_);
        for (size_t i = 0; i < dimension_; ++i)
        /*   if (GetPeriodic(i))
           {
               lower[i] = edges_.first[i] + spacing_[i] / 2;
           }
           else
          */
        {
            lower[i] = edges_.first[i];
        }
        return lower;
    }

    /**
     * @brief  Get the lower edge for a specific dimension.
     *
     * @param dim Index of the dimension.
     * \return Value of the lower edge in the requested dimension.
     *
     * \note The first dimension has the index 0.
     */
    float GetLower(size_t dim) const
    {
        if (dim >= GetDimension())
        {
            printf(
                "Warning! Lower edge requested for a dimension larger than the "
                "grid dimensionality!\n");
            return 0.0;
        }
        /* if (GetPeriodic(dim))
         {
             return edges_.first[dim] + 0.5* spacing_[i] ;
         }
         else
        */
        {
            return edges_.first[dim];
        }
    }

    /**
     * @brief /! Return the upper edges of the Grid.
     *
     * \return Vector containing the upper edges of the grid.
     */
    const vector<float> GetUpper() const
    {
        vector<float> upper(dimension_);
        for (size_t i = 0; i < dimension_; ++i)
        /*if (GetPeriodic(i))
        {
            upper[i] = edges_.second[i] - 0.5* spacing_[i];
        }
        else
      */ {
            upper[i] = edges_.second[i];
        }
        return upper;
    }

    /**
     * @brief Get the upper edge for a specific dimension.
     *
     * @param dim Index of the dimension.
     * \return Value of the upper edge in the given dimension.
     *
     * \note The dimensions are indexed starting with 0.
     */
    float GetUpper(size_t dim) const
    {
        if (dim >= GetDimension())
        {
            printf(
                "Warning! Upper edge requested for a dimension larger than the "
                "grid dimensionality!\n");
            return 0.0;
        }
        /*  if (GetPeriodic(dim))
          {
              return edges_.second[dim] - 0.5*spacing_[i];
          }
          else
          */
        {
            return edges_.second[dim];
        }
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

    /**
     * @brief  Return the Grid indices for a given point.
     *
     * @param x Point in space.
     * \return Indices of the grid point to which the point in space pertains.
     *
     * The grid discretizes the continuous space. For a given point in this
     * continuous space, this function will return the indices of the grid point
     * covering the point in space.
     *
     * If the grid is non-periodic in a given dimension and x is lower than the
     * lower edge in this dimension, the function will return -1, the index of
     * the underflow bin. Similarly, it will return numPoints, the index of the
     * overflow bin, if x is larger than the upper edge.
     *
     * In periodic dimensions, the index can take any integer value and will be
     * wrapped to the interval [0, numPoints).
     */
    vector<int> GetIndices(const vector<float>& x) const
    {
        // Check that input vector has the correct dimensionality
        if (x.size() != dimension_)
        {
            printf(
                "Specified point has a larger dimensionality than the grid.");
        }

        vector<int> indices(dimension_);
        for (size_t i = 0; i < dimension_; ++i)
        {
            float xpos = x.at(i);
            if (!GetPeriodic(i))
            {
                if (xpos < edges_.first[i])
                {
                    indices.at(i) = -1;
                    continue;
                }
                else if (xpos > edges_.second[i])
                {
                    indices.at(i) = numPoints_[i];
                    continue;
                }
            }

            // To make sure, the value is rounded in the correct direction.
            // spacing = (edges_.second[i] - edges_.first[i]) / (numPoints_[i]);
            indices.at(i) = floor((xpos - edges_.first[i]) * inv_spacing_[i]);
        }

        return wrapIndices(indices);
        // return indices;
    }

    /**
     * @brief /! Return the Grid index for a one-dimensional grid.
     * @param x Point in space.
     * \return Grid index to which the point pertains.
     *
     * Return the Grid index pertaining to the given point in space. This
     * function is for convenience when accessing 1d-Grids. For
     * higher-dimensional grids, x needs to be a vector of floats.
     */
    int GetIndex(float x) const
    {
        if (dimension_ != 1)
        {
            printf(
                "1d Grid index can only be accessed for 1d-Grids can be "
                "accessed with a.");
        }
        return GetIndices({x}).at(0);
    }

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

        vector<float> v(dimension_);

        for (size_t i = 0; i < dimension_; ++i)
        {
            // float spacing = (edges_.second[i] - edges_.first[i]) /
            // (numPoints_[i]);
            v.at(i) = edges_.first[i] + (indices[i] + 0.5) * spacing_[i];
        }

        return v;
    }

    /**
     * @brief  Return center point of 1d-grid
     * @param index Index of the 1d grid.
     * \return Coordinate in real/CV space.
     *
     * \note This function is only available for 1d grids.
     */
    float GetCoordinate(int index) { return GetCoordinates({index}).at(0); }

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
            printf(
                "Dimension of indices does not match dimension of the grid.");
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
            static_cast<const GridBase<T>*>(this)->at(indices));
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
    const T& at(int index) const
    {
        if (dimension_ != 1)
        {
            printf(
                "Only 1d-Grids can be accessed with a single integer as the "
                "index.");
        }
        return at({index});
    }

    /**
     * @brief  Access 1d Grid by index, read-write
     * @param index Index specifying the grid point.
     * \return Reference of value at the given grid point.
     *
     * \note This function can only be used for 1d-Grids.
     */
    T& at(int index)
    {
        return const_cast<T&>(static_cast<const GridBase<T>*>(this)->at(index));
    }

    /**
     * @brief  Access Grid element pertaining to a specific point -- read-only
     * @param x Vector of floats specifying a point.
     * \return Const reference of the value at the given coordinates.
     *
     * This function is provided for convenience. It is identical to
     * GridBase::at(GridBase::GetIndices(x)).
     */
    const T& at(const vector<float>& x) const { return at(GetIndices(x)); }

    /**
     * @brief  Access Grid element pertaining to a specific point -- read/write
     * @param x Vector of floats specifying a point.
     * \return Reference to the value at the given coordinates.
     *
     * This function is provided for convenience. It is identical to
     * GridBase::at(GridBase::GetIndices(x)).
     */
    T& at(const vector<float>& x) { return at(GetIndices(x)); }

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
        return const_cast<T&>(static_cast<const GridBase<T>*>(this)->at(x));
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
    virtual ~GridBase() {}

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
            /*
            if (index == numPoints_[i] - 1)
            {
                index = 0;
            }*/
            newIndices.at(i) = index;
        }

        return newIndices;
    }
    vector<T> data_;  ///< Internal storage of the data
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
     * classes of GridBase are constructed.
     */
    GridBase(const vector<int>& numPoints, const vector<float>& lower,
             const vector<float>& upper, const vector<bool>& isPeriodic)
        : dimension_(numPoints.size()),
          numPoints_(numPoints),
          edges_(pair<vector<float>, vector<float>>(lower, upper)),
          isPeriodic_(isPeriodic)
    {
        // Check that vector sizes are correct
        if (edges_.first.size() != dimension_ ||
            edges_.second.size() != dimension_)
        {
            printf(
                "Size of vector containing upper or lower edges, does not "
                "match size of vector containing "
                "number of grid points.");
        }
        if (isPeriodic_.size() == 0)  // Default: Non-periodic in all dimensions
        {
            isPeriodic_.resize(dimension_, false);
        }
        if (isPeriodic_.size() != dimension_)
        {
            printf(
                "Size of vector isPeriodic does not match size of vector "
                "containing number of grid points.");
        }
        for (size_t i = 0; i < isPeriodic_.size(); i++)
        {
            float spacing =
                (edges_.second[i] - edges_.first[i]) / (numPoints_[i]);
            spacing_.push_back(spacing);
            if (isPeriodic_[i])
            {
                if (numPoints_[i] <= 1)
                {
                    printf(
                        "A periodic grid is incompatible with a grid size of "
                        "1.");
                }
                float spacing =
                    (edges_.second[i] - edges_.first[i]) / (numPoints_[i] - 1);
            }
            else
            {
                numPoints_[i] += 1;
                edges_.first[i] -= spacing / 2;
                edges_.second[i] += spacing / 2;
            }
            inv_spacing_.push_back(1. / spacing_[i]);
        }
    }

    size_t dimension_;       ///< Dimension of the grid
    vector<int> numPoints_;  ///< Number of points in each dimension.
    pair<vector<float>, vector<float>>
        edges_;                  ///< Edges of the Grid in each dimension.
    vector<bool> isPeriodic_;    ///< Periodicity of the Grid.
    vector<float> spacing_;      ///< The Grid grain.
    vector<float> inv_spacing_;  ///< The Grid grain inverted.
};

#endif  // __GRIDBASE_H__
