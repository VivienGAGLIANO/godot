#ifndef __SAMPLER__H__
#define __SAMPLER__H__

#include <iostream>
#include <fstream>
#include <memory.h>

#include <random>
#include <ctime>
#include "image_vector.h"
#include "core/math/random_number_generator.h"

namespace TexSyn
{

//TODO: add iterator for samplers

/**
 * @brief The SamplerBase class is an interface for every Samplers.
 * the main idea is for them to provide an array of coordinates between (0, 0) and (1, 1),
 * and setters to modify the parameters of the generative process.
 */
class SamplerBase
{
public:

	using Vec2 = Vector2;
	using VectorType = std::vector<Vec2>;

	SamplerBase() {}
	virtual ~SamplerBase() {}

	virtual void generate(VectorType &vector, unsigned int nbPoints)=0;
	virtual Vec2 next()=0;
};

/**
 * @brief The SamplerOrigin class is an override of SamplerBase,
 * supposed to yield an array of points all located at the origin.
 */
class SamplerOrigin : public SamplerBase
{
public:
	SamplerOrigin();

	void generate(VectorType &vector, unsigned int nbPoints);
	Vec2 next();
};

/**
 * @brief The SamplerUniform class is an override of SamplerBase,
 * supposed to yield an array of points in respect to a uniform random process.
 */
class SamplerUniform : public SamplerBase
{
public:

	SamplerUniform(uint64_t seed = 0);

	void generate(VectorType &vector, unsigned int nbPoints);
	Vec2 next();

private:
	RandomNumberGenerator m_rand;
};


/**
 * @brief The SamplerCycles class is an override of SamplerBase,
 * supposed to yield an array of points in respect to a uniform random process.
 */
class SamplerPeriods : public SamplerBase
{
public:

	SamplerPeriods(uint64_t seed = 0);

	void setPeriods(const Vec2 &firstPeriod, const Vec2 &secondPeriod);

	void generate(VectorType &vector, unsigned int nbPoints);
	Vec2 next();

private:
	Vec2					m_periods[2];
	RandomNumberGenerator	m_rand;
	unsigned int			m_periodDenominator;
};

/**
 * @brief The SamplerCycles class is an override of SamplerBase,
 * supposed to yield an array of points in respect to a uniform random process.
 */
class SamplerImportance : public SamplerBase
{
public:

	using ImageScalarType = ImageScalar<float>;
	using ImageVectorType = ImageVector<float>;
	using ScalarVectorType = std::vector<float>;

	SamplerImportance(const ImageScalarType &importanceFunction, uint64_t seed=0);
	~SamplerImportance();

	/**
	 * @brief generate yields an array of floating point coordinates
	 * distributed with a probability relative to the importance map provided.
	 * @return the coordinates array.
	 */
	void generate(VectorType &vector, unsigned int nbPoints);
	Vec2 next();

	const ImageScalarType &importanceFunction() const;

private:

	template <typename Predicate>
	static int findInterval(int size, const Predicate &pred);

	//Classes heavily inspired from the course of PBRT on importance sampling.
	struct Distribution1D
	{
		Distribution1D(const float *f, int n);
		int count() const;
		float sampleContinuous(float u, float *pdf, int *off = nullptr) const;
		int sampleDiscrete(float u, float *pdf = nullptr, float *uRemapped = nullptr) const;
		float discretePDF(int index) const;

		ScalarVectorType m_func, m_cdf;
		float m_funcInt;
	};

	class Distribution2D
	{
	public:

		using PtrDistribution1DVectorType = std::vector<std::unique_ptr<Distribution1D>>;

		Distribution2D() {}
		void init(const float *func, int nu, int nv);
		Vec2 sampleContinuous(const Vec2 &u, float *pdf) const;
		float pdf(const Vec2 &p) const;

	private:
		PtrDistribution1DVectorType m_pConditionalV;
		std::unique_ptr<Distribution1D> m_pMarginal;
	};

private:
	Distribution2D			m_distribution2D;
	ImageScalarType			m_importanceFunction;
	RandomNumberGenerator	m_rand;
};

template<typename Predicate>
int SamplerImportance::findInterval(int size, const Predicate &pred)
{
	int first = 0, len = size;
	while (len > 0)
	{
		int half = len >> 1, middle = first + half;
		if (pred(middle))
		{
			first = middle + 1;
			len -= half + 1;
		}
		else
		{
			len = half;
		}
	}
	return CLAMP(first - 1, 0, size - 2);
}

} //namespace Stamping

#endif //__SAMPLER__H__
