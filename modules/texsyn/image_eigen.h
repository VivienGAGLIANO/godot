#ifndef TEXSYN_IMAGE_EIGEN_H
#define TEXSYN_IMAGE_EIGEN_H

#include "image_vector.h"
#include "eigen/Eigen/Dense"

namespace TexSyn
{

template<typename T>
void fromImageVectorToMatrix(const ImageVector<T> &image, Eigen::MatrixXd &matrix)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (const typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (const typename ImageVector<T>::DataType &pix)
		{
			matrix(j, i) = pix;
			++j;
		});
		++i;
	});
}

template<typename T>
void fromMatrixToImageVector(const Eigen::MatrixXd &matrix, ImageVector<T> &image)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (typename ImageVector<T>::DataType &pix)
		{
			pix = matrix(j, i);
			++j;
		});
		++i;
	});
}

};

#endif
