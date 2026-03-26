/**
 * @file
 * @author
 * - Zhijun Pan
 */

#ifndef __SWITCHFUCNTION_H__
#define __SWITCHFUCNTION_H__

using namespace std;
/**
 * @brief The continuous function mimicing
 *        naturely discrete CVs.
 * @details
 * Switching functions are useful for a variety of collective variables.
 * Several classes of these functions exist.
 *
 * Gaussian: Delta-like, for metadynamics.
 * Rational: Heaviside step like, for CN,NN.
 * Smap: a generalized sigmoid curve for sketch-map.
 * Q: native contacts (amino acid)
 * Cosinus: a generalized Bond Order Parameters for solid/liquid.
 */
class SwitchFunction
{
   public:
    /**
     * @brief Evaluate the pairwise switch function.
     * @param[in] rij distance between two atoms.
     * @param[out] df Reference variable storing gradient.
     *
     * \return value of pairwise switch function.
     */
    virtual float Evaluate(const float& rij, float& df) const = 0;

    /**
     * @brief  Build SwitchFunction from JSON value.
     * @param[in] json JSON value node.
     *
     * \return Pointer to new SwitchFunction.
     */
    // static SwitchFunction* Build(const Value& json);

    //! Destructor
    virtual ~SwitchFunction(void);
};
/**
 * @brief  Gaussian Function
 *
 * @details
 * A standard Gaussian function (also called a "bell curve").
 * \f[
 * s(r)=\exp\left(-\frac{ (r - d_0)^2 }{ 2r_0^2 }\right)
 * \f]
 */
class GaussianSF : public SwitchFunction
{
   public:
    /**
     * @brief Construct a Gaussian Switch Function.
     * @param[in] d0 Center of Gaussian.
     * @param[in] inv_r0 Width of Gaussian.
     */
    GaussianSF(const float& d0, const float& inv_r0, const float& tperiod);
    GaussianSF(const float& d0, const float& inv_r0);
    /**
     * @brief Construct a new GaussianSF
     *       using default parameter as: d0=0, r0=1.6
     */
    GaussianSF(void);
    float Evaluate(const float& rij, float& df) const override;
    const float& GetCenter(void) const;
    const float& GetWidth(void) const;

   private:
    float d0;      ///< Center of Gaussian.
    float inv_r0;  ///< Width of Gaussian.
    float period;  ///< period of Gaussian.
};

/**
 * @brief  Rational Switching Function
 * @details
 * A switching function of a rational form.
 * \f[
 *  s(r)=\frac{ 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{n} }
 *  { 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{m} }
 * \f]
 *
 */
class RationalSF : public SwitchFunction
{
   public:
    /**
     * @brief  Constructor.
     * @param[in] d0 Minimum linear shift value.
     * @param[in] inv_r0 Cutoff distance, inversion make it faster.
     * @param[in] m An exponent of the switching function.
     * @param[in] n An exponent of the switching function.
     *
     * Construct a Rational Switch Function.
     *
     */
    RationalSF(const float& d0, const float& inv_r0, const int& n,
               const int& m);
    RationalSF(const float& d0, const float& inv_r0);
    RationalSF(void);

    float Evaluate(const float& rij, float& df) const override;

    // static RationalSF* Build(const Value& json);

   private:
    float d0;      ///< Minimum linear shift value.
    float inv_r0;  ///< Cutoff distance.
    int n;         ///< Exponent of numerator in switching function (controls
                   ///< stiffness).
    int m;         ///< Exponent of denominator in switching function (controls
                   ///< stiffness).
};

/**
 * @brief  Sketch map
 *
 * @details
 * Sketch map based on Sigmoid switching Function
 * \f[
 * s(r) = \left[ 1 + ( 2^{a/b} -1 )\left( \frac{r-d_0}{r_0} \right)^a
 * \right]^{-b/a}
 * \f]
 * note: when a=b, this Smap function become rational function m=2n=2a.
 */
class SmapSF : public SwitchFunction
{
   public:
    /**
     * @brief Construct a Smap Switch Function.
     * @param[in] d0 Minimum linear shift value.
     * @param[in] r0 reference structure
     * @param[in] a  Exponent of numerator in switching function (controls
     * stiffness).
     * @param[in] b  Exponent of denominator in switching function (controls
     * stiffness).
     * @param[in] c  2^{a/b} -1
     * @param[in] d  -b/a
     */
    SmapSF(const float& td0, const float& tinv_r0, const float& ta,
           const float& tb, const float& tc, const float& td);
    /**
     * @brief Construct a new SmapSF
     * default parameter is d0(0.0), inv_r0(0.625), a(1.0), b(1.0), c(1.0),
     * d(-1.0)
     *
     */
    SmapSF(void);

    float Evaluate(const float& rij, float& df) const override;

   private:
    float d0;      ///< Center of smap
    float inv_r0;  ///< Width of smap
    float a;       ///< An exponent of the switching function.
    float b;       ///< An exponent of the switching function.
    float c;       ///< 2^{a/b} -1
    float d;       ///< -b/a
};

/**
 * @brief  Protein folding degree Q
 *
 * @details
 *  fraction of native tertiary contacts function  Q
 * \f[
 * s(r) = \frac{1}{1 + \exp(\beta(r_{ij} - \lambda r_{0}))}
 * \f]
 */
class QSF : public SwitchFunction
{
   public:
    /**
     * @brief Construct a Q Switch Function.
     * @param[in] beta = 5.0;  // A^{-1}
     * @param[in] lambda = 1.8; // unitless
     * @param[in] ref reference structure
     */
    QSF(const float& beta, const float& lambda, const float& ref);
    /**
     * @brief Construct a Q Switch Function using default parameter.
     *  beta = 5.0; lambda = 1.8; ref = 1.0.
     */
    QSF(void);
    float Evaluate(const float& rij, float& df) const override;

   private:
    float beta;  ///< beta = 5.0
    float r0;    ///<  r0 = lambda * ref
};

/**
 * @brief COSINUS Function: continuious cosin between 0,1.
 *
 * @details
 * To calculate Steinhardt bond orientation order parameter
 * Q^{\alpha\beta}_{l}.
 * \f[
 * s(r) &= 1 & if r<=d_0
 * s(r) &= 0.5 \left( \cos ( \frac{ r - d_0 }{ r_0 } * PI ) + 1 \right) & if
 * d_0<r<=d_0+r_0 s(r) &= 0 & if r> d_0+r_0
 * \f]
 */
class CosinusSF : public SwitchFunction
{
   public:
    /**
     * @brief Construct a Cosinus Switch Function.
     * @param[in] d0 Center of Cosinus.
     * @param[in] inv_r0 Width of Cosinus.
     */
    CosinusSF(const float& d0, const float& inv_r0);
    /**
     * @brief Construct a new CosinusSF
     *       using default parameter as: d0=0, r0=1.6
     */
    CosinusSF(void);
    float Evaluate(const float& rij, float& df) const override;

   private:
    float d0;      ///< Center of Cosinus.
    float inv_r0;  ///< Width of Cosinus.
};

#endif  // __SWITCHFUCNTION_H__
