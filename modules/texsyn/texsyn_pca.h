#ifndef TEXSYN_PCA_H
#define TEXSYN_PCA_H

#include "texsyn_statistics.h"

namespace TexSyn
{

template<typename T>
class PCA
{
public :
	using DataType	= T;
	using ImageType = ImageVector<DataType>;
	using VectorType = LocalVector<DataType>;
	using MatrixType = Matrix<DataType>;

	PCA (const ImageType& input);

	void mat_mult(double vin[3], double vout[3]);

	void computeEigenVectors();

	void computePCA();

	void project(ImageType &res);

	void back_project(const ImageType &coord, ImageType& res) const;

	VectorType eigenVector(unsigned int i);

private:

	const ImageType &m_im;

	// eigenvectors
//	double m_v1 [3];
//	double m_v2 [3];
//	double m_v3 [3];
	MatrixType m_eigenVectors;
	// mean values
	StatisticsVector<DataType> m_statistics;
	// covariance matrix
	MatrixType m_A_inverse; // inverse transpose
	DataType m_det;
	MatrixType m_A;

	double dot(const VectorType &v, const VectorType &w);
	DataType normalize(VectorType &v);
	void cross(const VectorType &v, const VectorType &w, VectorType &out);
};

}

#endif
