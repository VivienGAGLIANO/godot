#ifndef TEXSYN_PCA_H
#define TEXSYN_PCA_H

#include "texsyn_statistics.h"
#include "eigen/Eigen/Dense"

namespace TexSyn
{

template<typename T>
class PCA
{
public :
	using DataType	= T;
	using ImageType = ImageVector<DataType>;
	using ImageScalarType = typename ImageType::ImageScalarType;
	using VectorType = Eigen::VectorXd;
	using MatrixType = Eigen::MatrixXd;

	PCA (const ImageType& input);

	void computePCA(unsigned int nbComponents = 0);
	void project(ImageType &res);
	void back_project(const ImageType &input, ImageType& output) const;

	static void fromImageToMatrix(const ImageType &image, MatrixType &matrix);
	static void fromMatrixToImage(const MatrixType &matrix, ImageType &image);

private:

	void computeEigenVectors();

	// eigenvectors
	VectorType m_eigenValues;
	MatrixType m_eigenVectors;

	MatrixType m_matrix;
	MatrixType m_projection;
	VectorType m_mean;
};

template<typename T>
PCA<T>::PCA(const ImageType &input) :
	m_eigenValues(),
	m_eigenVectors(),
	m_matrix(),
	m_projection(),
	m_mean()
{
	//bug here
	m_matrix.resize(input.get_width() * input.get_height(), input.get_nbDimensions());
	fromImageToMatrix(input, m_matrix);
	computeEigenVectors();
}

template<typename T>
void PCA<T>::computeEigenVectors()
{
	// Subtract the mean from each column
	m_mean = m_matrix.colwise().mean();
	MatrixType centered = m_matrix.rowwise() - m_mean.transpose();

	// Calculate the correlation matrix
	MatrixType cov = centered.adjoint() * centered / double(centered.rows() - 1);

	// Compute the eigenvectors and eigenvalues of the covariance matrix
	Eigen::SelfAdjointEigenSolver<MatrixType> eig(cov);
	m_eigenValues = eig.eigenvalues().reverse();
	m_eigenVectors = eig.eigenvectors().rowwise().reverse();

	// Normalize the eigenvectors
	for (int i = 0; i < m_eigenVectors.cols(); i++)
	{
		double norm = m_eigenVectors.col(i).norm();
		m_eigenVectors.col(i) /= norm;
	}
}

template<typename T>
void PCA<T>::computePCA(unsigned int nbComponents)
{
	if(nbComponents == 0)
	{
		nbComponents = m_matrix.cols();
	}
	m_projection = m_matrix.rowwise() - m_mean.transpose();
	m_projection *= m_eigenVectors.leftCols(nbComponents);
}

template<typename T>
void PCA<T>::project(ImageType &res)
{
	fromMatrixToImage(m_projection, res);
}

template<typename T>
void PCA<T>::back_project(const ImageType &input, ImageType &output) const
{
	MatrixType matrix(input.get_width() * input.get_height(), input.get_nbDimensions());
	MatrixType projection = m_eigenVectors * m_eigenValues.asDiagonal() * m_eigenVectors.transpose();

	// Project the matrix back onto the original space
	MatrixType matrix_inv = matrix * projection.transpose() + m_mean.transpose();
	fromMatrixToImage(matrix_inv, output);
}

template<typename T>
void PCA<T>::fromImageToMatrix(const ImageType &image, MatrixType &matrix)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (const ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (const DataType& pix)
		{
			matrix(j, i) = pix;
			++j;
		});
		++i;
	});
}

template<typename T>
void PCA<T>::fromMatrixToImage(const MatrixType &matrix, ImageType &image)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (DataType& pix)
		{
			pix = matrix(j, i);
			++j;
		});
		++i;
	});
}

}

#endif
