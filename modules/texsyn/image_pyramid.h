#ifndef TEXSYN_IMAGE_PYRAMID_H
#define TEXSYN_IMAGE_PYRAMID_H

#include "core/object/ref_counted.h"
#include "image_vector.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#define TEXSYN_ASSERT_DEPTH(depth) 				DEV_ASSERT(is_initialized() && (depth) < get_depth())
#define TEXSYN_ASSERT_IMAGE_SQUARE(image) 		DEV_ASSERT((image).get_width() == (image).get_height())
#define TEXSYN_ASSERT_IMAGE_POWER_2(image) 		DEV_ASSERT((image).get_width()%2 == 0 && (image).get_height()%2 == 0)
#define TEXSYN_ASSERT_IMAGE_DEPTH(image) 		DEV_ASSERT(depth <= log2((image).get_width())+1\
										 		&& depth <= log2((image).get_height())+1)
#define TEXSYN_ASSERT_PYRAMID_INITIALIZED()     DEV_ASSERT(is_initialized())
#define TEXSYN_ASSERT_RANGE(alpha, beta)		DEV_ASSERT((alpha) <= (beta) && (beta) < this->get_depth())


// TODO extract Godot interfaced classes from here and put them in different file (at least out of namespace, and move definition to .cpp file)

namespace TexSyn {

/*****************************************************************************************************/
/****************************************** Backend classes ******************************************/
/*****************************************************************************************************/


/******************************************* Image pyramid *******************************************/

template <typename T, typename element>
class ImagePyramid
{
public:
	using DataType=T;
    using PyramidElement=element;
	using ImageVectorType=ImageVector<DataType>;

	ImagePyramid();
	ImagePyramid(int depth);
	ImagePyramid(const ImagePyramid &other);

	virtual ~ImagePyramid() = 0;

	bool is_initialized() const;
	int get_depth() const;

	PyramidElement get_layer(int depth) const;
	const PyramidElement &get_layerRef(int depth) const;
    PyramidElement &get_layerRef(int depth);

    ImagePyramid<DataType, PyramidElement> &operator=(const ImagePyramid<DataType, PyramidElement> &other);

	static ImageVectorType expand(const ImageVectorType &image);
	static ImageVectorType reduce(const ImageVectorType &image);

protected:
	int m_depth;
	std::vector<PyramidElement> m_layers;
};


/// Default constructor. Depth initialized at 0.
template <typename T, typename element>
ImagePyramid<T, element>::ImagePyramid() :
	m_depth(0),
	m_layers()
{}

/// Depth initializing constructor.
/// \param depth Chosen depth for pyramid. Includes input image.
template <typename T, typename element>
ImagePyramid<T, element>::ImagePyramid(int depth) :
	m_depth(depth),
	m_layers()
{}

/// Copy constructor. Layers are duplicated in memory.
/// \param other Pyramid to copy.
template <typename T, typename element>
ImagePyramid<T, element>::ImagePyramid(const ImagePyramid &other) :
	m_depth(other.get_depth()),
	m_layers(other.m_layers)
{
	print_line("Base copy constructor.");
}

/// Default destructor.
template <typename T, typename element>
ImagePyramid<T, element>::~ImagePyramid()
{}

/// Checks if pyramid is initialized.
/// \return Returns true if pyramid isn't empty.
template <typename T, typename element>
bool ImagePyramid<T, element>::is_initialized() const
{
	return m_layers.size() > 0;
}

template <typename T, typename element>
int ImagePyramid<T, element>::get_depth() const
{
	return m_depth;
}

/// Get selected pyramid layer.
/// \param depth must be less than pyramid depth.
/// \return Returns a copy of selected layer.
template<typename T, typename element>
typename ImagePyramid<T, element>::PyramidElement ImagePyramid<T, element>::get_layer(int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);
	return m_layers[depth];
}

/// Get reference to selected pyramid layer.
/// \param depth must be less than pyramid depth.
/// \return Returns a const reference to selected layer.
template<typename T, typename element>
const typename ImagePyramid<T, element>::PyramidElement &ImagePyramid<T, element>::get_layerRef(int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);
	return m_layers[depth];
}

/// Get reference to selected pyramid layer.
/// \param depth must be less than pyramid depth.
/// \return Returns a reference to selected layer.
template<typename T, typename element>
typename ImagePyramid<T, element>::PyramidElement &ImagePyramid<T, element>::get_layerRef(int depth)
{
	TEXSYN_ASSERT_DEPTH(depth);
	return const_cast<PyramidElement &>(get_layerRef(depth));
}

/// Copy assignment operator.
/// \return Returns instance so that operator can be chained.
template<typename T, typename element>
ImagePyramid<T, element> &ImagePyramid<T, element>::operator=(const ImagePyramid<DataType, PyramidElement> &other)
{
	m_depth = other.get_depth();
	m_layers = other.m_layers;

	return *this;
}

/// \brief Expand operation to upsize input image.
/// \details Performs smoothing and upsampling on input image. Even-numbered 0-valued rows and columns are added, and a 5x5 gaussian kernel is then used for smoothing.
/// \param image Image to upsize
/// \return Returns a blurred image that has the same format as input, but size is doubled in both width and length.
template <typename T, typename element>
typename ImagePyramid<T, element>::ImageVectorType ImagePyramid<T, element>::expand(const ImageVectorType &image)
{
	ImageVectorType i;
	i.init(image.get_width()*2, image.get_height()*2, image.get_nbDimensions(), true);

	for (int x=0; x<image.get_width(); ++x)
		for (int y=0; y<image.get_height(); ++y)
			for (unsigned int d=0; d<i.get_nbDimensions(); ++d)
			{
                auto pix = image.get_pixel(x,y,d);
                i.set_pixel(2*x, 2*y, d, pix);
                i.set_pixel(2*x+1, 2*y, d, pix);
                i.set_pixel(2*x, 2*y+1, d, pix);
                i.set_pixel(2*x+1, 2*y+1, d, pix);
			}

//  This is a partial fix to the bleeding edge problem. Not using the smoothing kernel, and using GL_NEAREST sampling, allows for a good (yet aliased) reconstruction. This will do for now.
//  Full solution would imply working on better texture packing and how to filter packed textures.

//	ImageScalar<DataType> kernel;
//	kernel.init(5, 5, false);
//	std::vector<DataType> v = {1, 4, 6, 4, 1};
//	auto func = [v](DataType &pix, int i, int j)
//	{
//		pix = v[i]*v[j]/static_cast<T>(64);
//	};
//	kernel.parallel_for_all_pixels(func);
//
//	i ^= kernel;

	return i;
}

/// \brief Reduce operation to downsize input image.
/// \details Performs smoothing and subsampling on input image. A 5x5 gaussian kernel is used for smoothing, and even-numbered rows and columns are then removed.
/// \param image Image to downsize
/// \return Returns a blurred image that has the same format as input, but size is halved in both width and length.
template<typename T, typename element>
typename ImagePyramid<T, element>::ImageVectorType ImagePyramid<T, element>::reduce(const ImageVectorType &image)
{
	ImageScalar<DataType> kernel;
	kernel.init(5, 5, false);
	std::vector<DataType> v = {1, 4, 6, 4, 1};
	auto func = [v](DataType &pix, int i, int j)
	{
		pix = v[i]*v[j]/static_cast<T>(256);
	};
	kernel.parallel_for_all_pixels(func);

	ImageVectorType blurred;
	blurred = image;
	blurred ^= kernel;

	ImageVectorType i;
	i.init(blurred.get_width()/2, blurred.get_height()/2, blurred.get_nbDimensions(), false);

	for (int x=0; x<i.get_width(); ++x)
		for (int y=0; y<i.get_height(); ++y)
			for (unsigned int d=0; d<i.get_nbDimensions(); ++d)
			{
				i.set_pixel(x, y, d, blurred.get_pixel(2*x+1, 2*y+1, d));
			}

	return i;
}


/****************************************** Gaussian pyramid *****************************************/

template<typename T>
class GaussianPyramid : public ImagePyramid<T, ImageVector<T>>
{
	using DataType=T;
	using ImageVectorType=ImageVector<DataType>;

public:
	GaussianPyramid(); // TODO remove the shit out of this if Godot classes can be made without default constructor
	GaussianPyramid(const ImageVectorType &image, int depth);

protected:
	void init(const ImageVectorType &image, int depth);
};


/// Default constructor. Depth initialized at 0.
template<typename T>
GaussianPyramid<T>::GaussianPyramid():
	ImagePyramid<T, ImageVector<T>>()
{}

/// Create gaussian pyramid from input image with desired depth.
/// \param image Input image size has to be a power of 2
/// \param depth Number of layers in pyramid (includes original image). Depth must verify depth <= log_2(min(width, length))+1 because each downsize operation halves input size.
template<typename T>
GaussianPyramid<T>::GaussianPyramid(const ImageVectorType &image, int depth):
	ImagePyramid<T, ImageVector<T>>(depth)
{
	init(image, depth);
}

/// Initialize gaussian pyramid by repeatedly downsizing input image until desired depth.
/// \param image Input image size has to be a power of 2
/// \param depth Number of layers in pyramid (includes original image). Depth must verify depth <= log_2(min(width, length))+1 because each downsize operation halves input size.
template <typename T>
void GaussianPyramid<T>::init(const GaussianPyramid::ImageVectorType &image, int depth)
{
	TEXSYN_ASSERT_IMAGE_POWER_2(image);
	TEXSYN_ASSERT_IMAGE_DEPTH(image);

	this->m_layers.reserve(depth);
	ImageVectorType img = image;
	this->m_layers.push_back(img);
	for (int i=1; i<depth; ++i)
	{
		img = this->reduce(img);
		this->m_layers.push_back(img);
	}
}

/***************************************** Laplacian pyramid *****************************************/

template<typename T>
class LaplacianPyramid : public ImagePyramid<T, ImageVector<T>>
{
	using DataType=T;
	using ImageVectorType=ImageVector<DataType>;

public:
	LaplacianPyramid();
	LaplacianPyramid(const ImageVectorType &image, int depth);
	LaplacianPyramid(const GaussianPyramid<DataType> &pyramid);

protected:
	void init(const GaussianPyramid<DataType> &pyramid);
};


/// Default constructor. Depth initialized at 0.
template <typename T>
LaplacianPyramid<T>::LaplacianPyramid() :
	ImagePyramid<T, ImageVector<T>>()
{}

/// Create laplacian pyramid from input image with desired depth.
/// \param image Input image size has to be a power of 2
/// \param depth Number of layers in pyramid (includes original image). Depth must verify depth <= log_2(min(width, length))+1 because each downsize operation halves input size.
template <typename T>
LaplacianPyramid<T>::LaplacianPyramid(const LaplacianPyramid::ImageVectorType &image, int depth) :
	LaplacianPyramid(GaussianPyramid<DataType>(image, depth))
{}

/// Create laplacian pyramid from gaussian pyramid.
/// \param pyramid Gaussian pyramid to use to compute laplacian layers.
template <typename T>
LaplacianPyramid<T>::LaplacianPyramid(const GaussianPyramid<DataType> &pyramid) :
	ImagePyramid<T, ImageVector<T>>(pyramid.get_depth())
{
	init(pyramid);
}

/// Initialize pyramid from gaussian pyramid. Each layer equals to corresponding gauss layer, minus the subsequent layer that has been expanded.
/// \param pyramid Gaussian pyramid to use to compute laplacian layers.
template <typename T>
void LaplacianPyramid<T>::init(const GaussianPyramid<DataType> &pyramid)
{
	this->m_layers.reserve(this->m_depth);
	ImageVectorType img = pyramid.get_layer(this->m_depth-1);
	this->m_layers.push_back(img);
	for (int i = 1; i < this->m_depth; ++i)
	{
		img = this->expand(pyramid.get_layer(this->m_depth-i));
		this->m_layers.push_back(pyramid.get_layer(this->m_depth-i-1) - img);
	}

	std::reverse(this->m_layers.begin(), this->m_layers.end());
}


/******************************************* Riesz pyramid *******************************************/

// TODO repplace Riesz pyramid construction method with better Riesz transform (cc Léo)

template <typename T>
class RieszPyramid;

template <typename T>
struct RieszLayer
{
	// Make some members private ? Is it useful ?

//    friend class RieszPyramid<T>;

//public:
    RieszLayer(const ImageVector<T> &image);
    RieszLayer(const RieszLayer<T> &layer);

    static void toPolar(const ImageVector<T> &fe, const ImageVector<T> &r1, const ImageVector<T> &r2, ImageVector<T> &amp, ImageVector<T> &pha, ImageVector<T> &ori);
    static void toCartesian(const ImageVector<T> &amp, const ImageVector<T> &pha, const ImageVector<T> &ori, ImageVector<T> &fe, ImageVector<T> &r1, ImageVector<T> &r2);

    RieszLayer<T> &operator=(const RieszLayer<T> &other);

//private:
    static void rieszTransform(const ImageVector<T> &image, ImageVector<T> &r1, ImageVector<T> &r2);

    ImageVector<T> fe, r1, r2, amp, pha, ori; // TODO store as smart pointer or data value ?
};


template <typename T>
RieszLayer<T>::RieszLayer(const ImageVector<T> &image)
{
    fe = image;

    rieszTransform(fe, r1, r2);
    toPolar(fe, r1, r2, amp, pha, ori);
}

template <typename T>
RieszLayer<T>::RieszLayer(const RieszLayer<T> &layer) :
    fe(layer.fe), r1(layer.r1), r2(layer.r2), amp(layer.amp), pha(layer.pha), ori(layer.ori)
{}

template <typename T>
void RieszLayer<T>::rieszTransform(const ImageVector<T> &image, ImageVector<T> &r1, ImageVector<T> &r2)
{
    ImageScalar<T> kernel;
    kernel.init(5, 5, true);
    kernel.set_pixel(0, 1, static_cast<T>(.5));
    kernel.set_pixel(2, 1, static_cast<T>(-.5));
    r1 = image ^ kernel;

	kernel.init(5, 5, true);
	kernel.set_pixel(1, 2, static_cast<T>(.5));
    kernel.set_pixel(1, 0, static_cast<T>(-.5));
    r2 = image ^ kernel;
}

// TODO implement element-wise methods in ImageVector class (cos, sin, etc...)
template <typename T>
void RieszLayer<T>::toPolar(const ImageVector<T> &fe, const ImageVector<T> &r1, const ImageVector<T> &r2, ImageVector<T> &amp, ImageVector<T> &pha, ImageVector<T> &ori)
{
    amp = fe*fe + r1*r1 + r2*r2;
    amp.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = sqrt(pix); });
        }
    );

    ImageVector<T> im = r1*r1 + r2*r2;
    im.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = sqrt(pix); });
        }
    );
	pha.init(fe.get_width(), fe.get_height(), fe.get_nbDimensions(), false);
    pha.parallel_for_all_images
    (
        [&fe, &im](ImageScalar<T> &image, int channel)
        {
            image.parallel_for_all_pixels
            (
                [&fe, &im, channel](T &pix, int x, int y) { pix = atan2(im.get_pixel(x, y, channel), fe.get_pixel(x, y, channel)); }
            );
        }
    );

	ori.init(fe.get_width(), fe.get_height(), fe.get_nbDimensions(), false);
    ori.parallel_for_all_images
    (
        [&r1, &r2](ImageScalar<T> &image, int channel)
        {
            image.parallel_for_all_pixels
            (
                [&r1, &r2, channel](T &pix, int x, int y) { pix = atan2(r1.get_pixel(x, y, channel), r2.get_pixel(x, y, channel)); }
            );
        }
    );

}

template <typename T>
void RieszLayer<T>::toCartesian(const ImageVector<T> &amp, const ImageVector<T> &pha, const ImageVector<T> &ori, ImageVector<T> &fe, ImageVector<T> &r1, ImageVector<T> &r2)
{
    fe = pha;
    fe.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = cos(pix); });
        }
    );
    fe *= amp;

    ImageVector<T> sin_pha = pha;
    sin_pha.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = sin(pix); });
        }
    );

    ImageVector<T> cos_ori = ori;
    cos_ori.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = cos(pix); });
        }
    );
    r1 = amp * sin_pha * cos_ori;

    ImageVector<T> sin_ori = ori;
    sin_ori.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = sin(pix); });
        }
    );
    r2 = amp * sin_pha * sin_ori;
}

template <typename T>
RieszLayer<T> &RieszLayer<T>::operator=(const RieszLayer<T> &other)
{
    fe = other.fe;
    r1 = other.r1;
    r2 = other.r2;
    amp = other.amp;
    pha = other.pha;
    ori = other.ori;

	return *this;
}

template <typename T>
class RieszPyramid : public ImagePyramid<T, RieszLayer<T>>
{
	using DataType=T;
	using ImageVectorType=ImageVector<DataType>;


public:
    enum CoordType // TODO replace strings by enum in method args
    {
        CARTESIAN,
        POLAR
    };

	RieszPyramid(); // TODO remove this shit if possible. Else init(pyr) has to initialize properly all member variables
	RieszPyramid(const ImageVectorType &image, int depth);
	RieszPyramid(const LaplacianPyramid<DataType> &pyramid);

    std::vector<ImageVectorType> pack_in_texture(const CoordType &system = CoordType::CARTESIAN) const;
    ImageVectorType reconstruct() const;

protected:
	void init(const LaplacianPyramid<DataType> &pyramid);

private:
};


template <typename T>
RieszPyramid<T>::RieszPyramid() :
	ImagePyramid<T, RieszLayer<T>>()
{}

template <typename T>
RieszPyramid<T>::RieszPyramid(const ImageVectorType &image, int depth) :
	RieszPyramid<T>(LaplacianPyramid<T>(image, depth))
{}

template <typename T>
RieszPyramid<T>::RieszPyramid(const LaplacianPyramid<DataType> &pyramid) :
	ImagePyramid<T, RieszLayer<T>>(pyramid.get_depth())
{
    init(pyramid);
}

template <typename T>
std::vector<typename RieszPyramid<T>::ImageVectorType> RieszPyramid<T>::pack_in_texture(const CoordType &system) const
{
    std::vector<ImageVectorType> packed_textures(3);
	int width = 0,
        height = this->m_layers.at(0).fe.get_height();
    for (const auto &layer : this->m_layers)
        width += layer.fe.get_width();


    packed_textures[0].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);
    packed_textures[1].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);
    packed_textures[2].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);

    int x = 0;
    for (const auto &layer : this->m_layers)
    {
        switch (system)
        {
            case CoordType::CARTESIAN:
                packed_textures[0].set_rect(layer.fe, x, 0);
                packed_textures[1].set_rect(layer.r1, x, 0);
                packed_textures[2].set_rect(layer.r2, x, 0);

                break;

            case CoordType::POLAR:
                packed_textures[0].set_rect(layer.amp, x, 0);
                packed_textures[1].set_rect(layer.pha, x, 0);
                packed_textures[2].set_rect(layer.ori, x, 0);

                break;

            default:
                std::cout << "Unrecognized coordinate system." << std::endl;
        }

        x += layer.fe.get_width();
    }

    return packed_textures;
}

template <typename T>
typename RieszPyramid<T>::ImageVectorType RieszPyramid<T>::reconstruct() const
{
    TEXSYN_ASSERT_INITIALIZED();

    RieszLayer<T> layer = this->get_layer(this->m_depth-1);
    ImageVectorType fe = layer.amp * ImageVectorType::cos(layer.pha);
    if (this->m_depth == 1) return fe;

    for (int i = this->m_depth-2; i >= 0; --i)
    {
        layer = this->get_layer(i);
        fe = ImagePyramid<T, RieszLayer<T>>::expand(fe);
        fe += layer.amp * ImageVectorType::cos(layer.pha);
    }

    return fe;
}

template <typename T>
void RieszPyramid<T>::init(const LaplacianPyramid<DataType> &pyramid)
{
    for (int i = 0; i < pyramid.get_depth(); ++i)
    {
        const ImageVectorType &layer = pyramid.get_layerRef(i);
		this->m_layers.emplace_back( layer);
    }
}



/*****************************************************************************************************/
/************************************** Godot interface classes **************************************/
/*****************************************************************************************************/

/***************************************** Gaussian pyramid ******************************************/

class GaussianPyr : public RefCounted
{
	GDCLASS(GaussianPyr, RefCounted);

public:
	inline GaussianPyr();

	inline void init(const Ref<Image> &image, int depth);
	inline void init_from_pyramid(const Ref<GaussianPyr> &pyramid);
	inline Ref<Image> get_layer(int depth) const;

private:
	inline int get_depth() const;
	inline bool is_initialized() const;

	GaussianPyramid<double> pyr;


protected:
	static void _bind_methods();
};


GaussianPyr::GaussianPyr()
	: RefCounted(), pyr()
{}

void GaussianPyr::init(const Ref<Image> &image, int depth)
{
	ImageVector<double> img;
	img.fromImage(image);
	pyr = GaussianPyramid<double>(img, depth);
}

void GaussianPyr::init_from_pyramid(const Ref<GaussianPyr> &pyramid)
{
	pyr = GaussianPyramid<double>(pyramid->pyr);
}

int GaussianPyr::get_depth() const
{
	return pyr.get_depth();
}

// TODO select Image format and toImage scale dynamically
Ref<Image> GaussianPyr::get_layer(int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);

	auto layer = pyr.get_layer(depth);
	Ref<Image> img = Image::create_empty(layer.get_width(), layer.get_height(), true, Image::FORMAT_RGBF);

	layer.toImage(img);

	return img;
}

bool GaussianPyr::is_initialized() const
{
	return pyr.is_initialized();
}

/***************************************** Laplacian pyramid *****************************************/

class LaplacianPyr : public RefCounted
{
	GDCLASS(LaplacianPyr, RefCounted);

public:
	inline LaplacianPyr();

	inline void init(const Ref<Image> &image, int depth);
	inline void init_from_pyramid(const Ref<LaplacianPyr> &pyramid);
	inline Ref<Image> get_layer(int depth) const;

private:
	inline int get_depth() const;
	inline bool is_initialized() const;

	LaplacianPyramid<double> pyr;


protected:
	static void _bind_methods();

};

LaplacianPyr::LaplacianPyr() :
	pyr()
{}

void LaplacianPyr::init(const Ref<Image> &image, int depth)
{
	ImageVector<double> img;
	img.fromImage(image);
	pyr = LaplacianPyramid<double>(img, depth);
}

void LaplacianPyr::init_from_pyramid(const Ref<LaplacianPyr> &pyramid)
{
	pyr = LaplacianPyramid<double>(pyramid->pyr);
}

Ref<Image> LaplacianPyr::get_layer(int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);

	auto layer = pyr.get_layer(depth);
	Ref<Image> img = Image::create_empty(layer.get_width(), layer.get_height(), true, Image::FORMAT_RGBF);

	layer.toImage(img);

	return img;
}

int LaplacianPyr::get_depth() const
{
	return pyr.get_depth();
}

bool LaplacianPyr::is_initialized() const
{
	return pyr.is_initialized();
}


/******************************************* Riesz pyramid *******************************************/

class RieszPyr : public RefCounted
{
	GDCLASS(RieszPyr, RefCounted);

public:
	enum CoordType
	{
		CARTESIAN,
		POLAR
	};


public:
    inline RieszPyr();

    inline void init(const Ref<Image> &image, int depth);
	inline bool is_initialized() const;
	inline int get_depth() const;
	inline Dictionary get_layer(int depth, CoordType type = CoordType::POLAR);
	inline RieszPyramid<double> get_pyramid() const;
    inline Dictionary pack_in_texture(CoordType type = CoordType::CARTESIAN) const;
    inline Ref<Image> phase_congruency(int alpha, int beta, CoordType type = CoordType::CARTESIAN);
    inline Ref<Image> reconstruct() const;

	inline Ref<Image> test(const Ref<Image> &image, const Ref<Image> &subImage) const;

private:
    RieszPyramid<double> pyr;

protected:
	static void _bind_methods();
};


TexSyn::RieszPyr::RieszPyr() :
    pyr(RieszPyramid<double>())
{}

void RieszPyr::init(const Ref<Image> &image, int depth)
{
    ImageVector<double> img;
    img.fromImage(image);
    pyr = RieszPyramid<double>(img, depth);
}

bool RieszPyr::is_initialized() const
{
	return pyr.is_initialized();
}

int RieszPyr::get_depth() const
{
	return pyr.get_depth();
}

Dictionary RieszPyr::get_layer(int depth, CoordType type)
{
	RieszLayer<double> layer = pyr.get_layer(depth);
	int width = layer.fe.get_width(),
		height = layer.fe.get_height();
	Dictionary dict;


	switch (type)
	{
		case CoordType::CARTESIAN:
		{
			Ref<Image> fe = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> r1 = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> r2 = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			layer.fe.toImage(fe);
			layer.r1.toImage(r1);
			layer.r2.toImage(r2);
			fe->generate_mipmaps();
			r1->generate_mipmaps();
			r2->generate_mipmaps();
			dict["fe"] = fe;
			dict["r1"] = r1;
			dict["r2"] = r2;
			break;
		}

		case CoordType::POLAR:
		{
			Ref<Image> amp = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> pha = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> ori = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			layer.amp.toImage(amp);
			layer.pha.toImage(pha);
			layer.ori.toImage(ori);
			amp->generate_mipmaps();
			pha->generate_mipmaps();
			ori->generate_mipmaps();
			dict["amp"] = amp;
			dict["pha"] = pha;
			dict["ori"] = ori;
			break;
		}

		default:
			print_line("Unrecognized type, unable to fetch layer.");
			break;

	}

	return dict;
}

RieszPyramid<double> RieszPyr::get_pyramid() const
{
	return pyr;
}

Dictionary RieszPyr::pack_in_texture(CoordType type) const
{
    Dictionary dict;

    std::vector<ImageVector<double>> packed_textures;
	int width, height = pyr.get_layer(0).fe.get_height();

    switch (type)
    {
        case CoordType::CARTESIAN:
		{
            packed_textures = pyr.pack_in_texture(RieszPyramid<double>::CoordType::CARTESIAN);
            width = packed_textures[0].get_width();
			Ref<Image> fe = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> r1 = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> r2 = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
            packed_textures[0].toImage(fe);
            packed_textures[1].toImage(r1);
            packed_textures[2].toImage(r2);
            fe->generate_mipmaps();
            r1->generate_mipmaps();
            r2->generate_mipmaps();
            dict["fe"] = fe;
            dict["r1"] = r1;
            dict["r2"] = r2;
            break;
		}

        case CoordType::POLAR:
		{
            packed_textures = pyr.pack_in_texture(RieszPyramid<double>::CoordType::POLAR);
            width = packed_textures[0].get_width();
			Ref<Image> amp = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> pha = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
			Ref<Image> ori = Image::create_empty(width, height, true, Image::FORMAT_RGBF);
            packed_textures[0].toImage(amp);
            packed_textures[1].toImage(pha);
            packed_textures[2].toImage(ori);
            amp->generate_mipmaps();
            pha->generate_mipmaps();
            ori->generate_mipmaps();
            dict["amp"] = amp;
            dict["pha"] = pha;
            dict["ori"] = ori;
            break;
		}

        default:
            print_line("Unrecognized type, unable to fetch layer.");
            break;
    }

    return dict;
}

Ref<Image> RieszPyr::reconstruct() const
{
    Ref<Image> rec = Image::create_empty(pyr.get_layer(0).fe.get_width(), pyr.get_layer(0).fe.get_height(), true, Image::FORMAT_RGBF);

    pyr.reconstruct().toImage(rec);
	rec->generate_mipmaps();

    return rec;
}

Ref<Image> RieszPyr::test(const Ref<Image> &image, const Ref<Image> &subImage) const
{
	ImageVector<double> img;
	img.fromImage(image);
	ImageVector<double> subImg;
	subImg.fromImage(subImage);

	img.set_rect(subImg, 50, 50);

    Ref<Image> output = Image::create_empty(image->get_width(), image->get_height(), true, Image::FORMAT_RGBF);
    img.toImage(output);

    return output;
}

} //namespace TexSyn

VARIANT_ENUM_CAST(TexSyn::RieszPyr::CoordType);


#endif //TEXSYN_IMAGE_PYRAMID_H
