#include "texsyn_sampler.h"

namespace TexSyn
{

SamplerOrigin::SamplerOrigin():
	SamplerBase()
{}

void SamplerOrigin::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	return;
}

SamplerOrigin::Vec2 SamplerOrigin::next()
{
	return Vec2(0.0, 0.0);
}

SamplerUniform::SamplerUniform(uint64_t seed) :
	m_rand()
{
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

void SamplerUniform::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerUniform::Vec2 SamplerUniform::next()
{
	return Vec2(m_rand.randf(), m_rand.randf());
}

SamplerPeriods::SamplerPeriods(uint64_t seed) :
	m_periods(),
	m_rand(),
	m_periodDenominator(0)
{
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

void SamplerPeriods::setPeriods(const Vec2 &firstPeriod, const Vec2 &secondPeriod)
{
	m_periods[0] = firstPeriod;
	m_periods[1] = secondPeriod;
	m_periodDenominator = (m_periods[0][0] > 0.005 ? 1.0/m_periods[0][0] : 1.0)
		* (m_periods[0][1] > 0.005 ? 1.0/m_periods[0][1] : 1.0)
		* (m_periods[1][0] > 0.005 ? 1.0/m_periods[1][0] : 1.0)
		* (m_periods[1][1] > 0.005 ? 1.0/m_periods[1][1] : 1.0)-1;
}

void SamplerPeriods::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerPeriods::Vec2 SamplerPeriods::next()
{
	Vec2 v;
	double first = double(m_rand.randi_range(0, m_periodDenominator));
	double second = double(m_rand.randi_range(0, m_periodDenominator));
	double x = first * m_periods[0][0] + second * m_periods[1][0];
	double y = first * m_periods[0][1] + second * m_periods[1][1];
	v[0] = x-floor(x);
	v[1] = y-floor(y);
	return v;
}

SamplerImportance::SamplerImportance(const ImageScalarType &importanceFunction, uint64_t seed) :
	m_distribution2D(),
	m_importanceFunction(importanceFunction),
	m_rand()
{
	m_distribution2D.init(m_importanceFunction.get_vector().ptr(), importanceFunction.get_width(), importanceFunction.get_height());
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
}

SamplerImportance::~SamplerImportance()
{}

void SamplerImportance::generate(VectorType &vector, unsigned int nbPoints)
{
	vector.resize(nbPoints);
	for(Vec2 &v : vector)
	{
		v = next();
	}
	return;
}

SamplerImportance::Vec2 SamplerImportance::next()
{
	static float pdf;
	Vec2 base(m_rand.randf(), m_rand.randf());
	return m_distribution2D.sampleContinuous(base, &pdf);
}

const SamplerImportance::ImageScalarType &SamplerImportance::importanceFunction() const
{
	return m_importanceFunction;
}

SamplerImportance::Distribution1D::Distribution1D(const float *f, int n)
	: m_func(f, f + n), m_cdf(n + 1)
{
	m_cdf[0] = 0;
	for (int i = 1; i < n+1; ++i)
	{
		m_cdf[i] = m_cdf[i - 1] + m_func[i - 1] / n;
	}
	m_funcInt = m_cdf[n];
	if (m_funcInt == 0)
	{
		for (int i = 1; i < n+1; ++i)
		{
			m_cdf[i] = float(i) / float(n);
		}
	}
	else
	{
		for (int i = 1; i < n+1; ++i)
		{
			m_cdf[i] /= m_funcInt;
		}
	}
}
int SamplerImportance::Distribution1D::count() const
{
	return m_func.size();
}

float SamplerImportance::Distribution1D::sampleContinuous(float u, float *pdf, int *off) const
{
	int offset = findInterval(m_cdf.size(), [&](int index)
	{
		return m_cdf[index] <= u;
	});
	if (off != nullptr)
	{
		*off = offset;
	}
	float du = u - m_cdf[offset];
	if ((m_cdf[offset+1] - m_cdf[offset]) > 0)
	{
		du /= (m_cdf[offset+1] - m_cdf[offset]);
	}
	if (pdf != nullptr)
	{
		*pdf = m_func[offset] / m_funcInt;
	}
	return (offset + du) / count();
}

int SamplerImportance::Distribution1D::sampleDiscrete(float u, float *pdf, float *uRemapped) const
{
	int offset = findInterval(m_cdf.size(), [&](int index)
	{
		return m_cdf[index] <= u;
	});

	if (pdf != nullptr)
	{
		*pdf = m_func[offset] / (m_funcInt * count());
	}
	if (uRemapped != nullptr)
	{
		*uRemapped = (u - m_cdf[offset]) / (m_cdf[offset+1] - m_cdf[offset]);
	}
	return offset;
}

float SamplerImportance::Distribution1D::discretePDF(int index) const
{
	return m_func[index] / (m_funcInt * count());
}

void SamplerImportance::Distribution2D::init(const float *func, int nu, int nv)
{
	for (int v = 0; v < nv; ++v)
	{
		m_pConditionalV.emplace_back(new Distribution1D(&func[v*nu], nu));
	}
	ScalarVectorType marginalFunc;
	for (int v = 0; v < nv; ++v)
	{
		marginalFunc.push_back(m_pConditionalV[v]->m_funcInt);
	}
	m_pMarginal.reset(new Distribution1D(&marginalFunc[0], nv));
}


SamplerImportance::Vec2 SamplerImportance::Distribution2D::sampleContinuous(const Vec2 &u, float *pdf) const
{
	float pdfs[2];
	int v;
	float d1 = m_pMarginal->sampleContinuous(u[1], &pdfs[1], &v);
	float d0 = m_pConditionalV[v]->sampleContinuous(u[0], &pdfs[0]);
	*pdf = pdfs[0] * pdfs[1];
	return Vec2(d0, d1);
}

float SamplerImportance::Distribution2D::pdf(const Vec2 &p) const
{
	int iu = CLAMP(int(p[0] * m_pConditionalV[0]->count()), 0, m_pConditionalV[0]->count() - 1);
	int iv = CLAMP(int(p[1] * m_pMarginal->count()), 0, m_pMarginal->count() - 1);
	return m_pConditionalV[iv]->m_func[iu] / m_pMarginal->m_funcInt;
}

}
