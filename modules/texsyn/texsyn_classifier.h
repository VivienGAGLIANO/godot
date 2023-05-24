#ifndef TEXSYN_CLASSIFIER_H
#define TEXSYN_CLASSIFIER_H

#include "core/io/image.h"
#include "image_vector.h"
#include <vector>

namespace TexSyn
{

Color vector_to_color(const Vector &v)
{
	ERR_FAIL_COND_V_MSG(v.size() <= 0 || v.size() > 4, Color(), "vector must have between one and four channels.");

	Color c;
	switch (v.size())
	{
		case 4:
			c.a = v[3];
		case 3:
			c.b = v[2];
		case 2:
			c.g = v[1];
		case 1:
			c.r = v[0];
		default:
			break;
	}

	return c;
}

double color_squared_distance(const Color &c1, const Color &c2)
{
	Color dif = c1-c2;

	return dif*dif;
}


template <typename T>
class ClassifierBase
{
public:
	ClassifierBase();
//	ClassifierBase(const ImageVector<T> &image);

	virtual ImageScalar<int> classify(const ImageVector<T> &image, const std::vector<std::pair<int,int>> &initial_centers) = 0;

protected:
//	ImageVector<T> m_image;
};

template <typename T>
ClassifierBase<T>::ClassifierBase()
	: m_image()
{}

//template <typename T>
//ClassifierBase<T>::ClassifierBase(const ImageVector<T> &image)
//		: m_image(image)
//{}


template <typename T>
class ClassifierKMeans : public ClassifierBase<T>
{
public:
	ClassifierKMeans(int max_iterations);
//	ClassifierKMeans(const ImageVector<T> &image);

	virtual ImageScalar<int> classify(const ImageVector<T> &image, const Vector<Vector<int>> &initial_centers) override;

private:
	int m_K;
	int m_max_iterations;
};

template <typename T>
ClassifierKMeans<T>::ClassifierKMeans(int max_iterations)
	: ClassifierBase<T>(), m_K(), m_max_iterations(max_iterations)
{}

//template <typename T>
//ClassifierKMeans<T>::ClassifierKMeans(const ImageVector<T> &image)
//		: ClassifierBase<T>(image), m_n_clusters()
//{}

// initial_centers a vector of coordinates for the pixels to take as initial cluster centers
template <typename T>
ImageScalar<int> ClassifierKMeans<T>::classify(const ImageVector<T> &image, const Vector<Vector<int>> &initial_centers)
{
	ERR_FAIL_COND_V_MSG(image.ptr() == nullptr, "image must not be null.");
	ERR_FAIL_COND_V_MSG(image.get_nbDimensions() != 3, "image must have exactly 3 channels.");

	m_K = initial_centers.size();
	ImageScalar<int> regions;
	regions.init(image.get_width(), image.get_height(), true);
	Vector<Color> centers(initial_centers.size()); // centers a vector of cluster centers, i.e. the color mean of each region's pixels (as Color type)

	Vector::ConstIterator cit = initial_centers.begin();
	Vector::Iterator it = centers.begin()
	for (; cit != initial_centers.end(); ++cit, ++it)
		*it = vector_to_color(image.get_pixel((*cit)[0], (*cit)[1]));

	for (int i = 0; i < m_max_iterations; ++i)
	{
		for (int y = 0; y < image.get_height(); ++y)
		for (int x = 0; x < image.get_width(); ++x)
		{
			Color region_center = centers[regions.get_pixel(x,y)];
			Color pixel = vector_to_color(image.get_pixel(x,y));

			double min_distance = color_squared_distance(pixel, region_center);

			for (int k = 0; k < m_K; ++k)
			{
				Color current_center = centers[i];
				double distance = color_squared_distance(pixel, current_center);

				if (distance < min_distance) regions.set_pixel(x, y, k);
			}
		}

		for (int k = 0; k < m_K; ++k)
		{
			Color mean(0, 0, 0, 1);
			int count = 0;

			for (int y = 0; y < image.get_height(); ++y)
			for (int x = 0; x < image.get_width(); ++x)
				if (regions.get_pixel(x, y) == k)
				{
					mean += vector_to_color(imageget_pixel(x,y));
					++count;
				}

			centers[k] = mean / count;
		}
	}

	return regions;
}
} //namespace TexSyn

#endif //TEXSYN_CLASSIFIER_H