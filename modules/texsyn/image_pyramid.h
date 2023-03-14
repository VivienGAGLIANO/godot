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
	ImagePyramid(unsigned int depth);
	ImagePyramid(const ImagePyramid &other);

	virtual ~ImagePyramid() = 0;

	bool is_initialized() const;
	int get_depth() const;

	PyramidElement get_layer(int depth) const;
	const PyramidElement &get_layerRef(int depth) const;
    PyramidElement &get_layerRef(int depth);

    ImagePyramid<DataType, PyramidElement> &operator=(const ImagePyramid<DataType, PyramidElement> &other);

protected:
	static ImageVectorType expand(const ImageVectorType &image);
	static ImageVectorType reduce(const ImageVectorType &image);

	unsigned int m_depth;
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
ImagePyramid<T, element>::ImagePyramid(unsigned int depth) :
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

	for (unsigned int x=0; x<image.get_width(); ++x)
		for (unsigned int y=0; y<image.get_height(); ++y)
			for (unsigned int d=0; d<i.get_nbDimensions(); ++d)
			{
				i.set_pixel(2*x+1, 2*y+1, d, image.get_pixel(x, y, d));
			}

	ImageScalar<DataType> kernel;
	kernel.init(5, 5, false);
	std::vector<DataType> v = {1, 4, 6, 4, 1};
	auto func = [v](DataType &pix, int i, int j)
	{
		pix = v[i]*v[j]/64.;
	};
	kernel.parallel_for_all_pixels(func);

	i ^= kernel;

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
		pix = v[i]*v[j]/256.;
	};
	kernel.parallel_for_all_pixels(func);

	ImageVectorType blurred;
	blurred = image;
	blurred ^= kernel;

	ImageVectorType i;
	i.init(blurred.get_width()/2, blurred.get_height()/2, blurred.get_nbDimensions(), false);

	for (unsigned int x=0; x<i.get_width(); ++x)
		for (unsigned int y=0; y<i.get_height(); ++y)
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
	GaussianPyramid(const ImageVectorType &image, unsigned int depth);

protected:
	void init(const ImageVectorType &image, unsigned int depth);
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
GaussianPyramid<T>::GaussianPyramid(const ImageVectorType &image, unsigned int depth):
	ImagePyramid<T, ImageVector<T>>(depth)
{
	init(image, depth);
}

/// Initialize gaussian pyramid by repeatedly downsizing input image until desired depth.
/// \param image Input image size has to be a power of 2
/// \param depth Number of layers in pyramid (includes original image). Depth must verify depth <= log_2(min(width, length))+1 because each downsize operation halves input size.
template <typename T>
void GaussianPyramid<T>::init(const GaussianPyramid::ImageVectorType &image, unsigned int depth)
{
	TEXSYN_ASSERT_IMAGE_POWER_2(image);
	TEXSYN_ASSERT_IMAGE_DEPTH(image);

	this->m_layers.reserve(depth);
	ImageVectorType img = image;
	this->m_layers.push_back(img);
	for (unsigned int i=1; i<depth; ++i)
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
	LaplacianPyramid(const ImageVectorType &image, unsigned int depth);
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
LaplacianPyramid<T>::LaplacianPyramid(const LaplacianPyramid::ImageVectorType &image, unsigned int depth) :
	LaplacianPyramid(GaussianPyramid<DataType>(image, depth))
{}

/// Create laplacian pyramid from gaussian pyramid.
/// \param pyramid Gaussian pyramid to use to compute laplacian layers.
template <typename T>
LaplacianPyramid<T>::LaplacianPyramid(const GaussianPyramid<DataType> &pyramid) :
	ImagePyramid<T, ImageVector<T>>(pyramid.get_depth())
{
	print_line("Derived copy constructor.");
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
        [&fe, &im](ImageScalar<T> &image, unsigned int channel)
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
        [&r1, &r2](ImageScalar<T> &image, unsigned int channel)
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
	RieszPyramid(const ImageVectorType &image, unsigned int depth);
	RieszPyramid(const LaplacianPyramid<DataType> &pyramid);

    std::vector<ImageVectorType> pack_in_texture(const CoordType &system = CoordType::CARTESIAN) const;
	ImageVectorType phase_congruency(int alpha, int beta, const std::string &system = "cartesian") const;
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
RieszPyramid<T>::RieszPyramid(const ImageVectorType &image, unsigned int depth) :
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
    for (const auto layer : this->m_layers)
        width += layer.fe.get_width();


    packed_textures[0].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);
    packed_textures[1].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);
    packed_textures[2].init(width, height, this->m_layers[0].fe.get_nbDimensions(), true);

    int x = 0;
    for (const auto layer : this->m_layers)
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
typename RieszPyramid<T>::ImageVectorType RieszPyramid<T>::phase_congruency(int alpha, int beta, const std::string &system) const
{
	TEXSYN_ASSERT_RANGE(alpha, beta);

    DataType eps = static_cast<DataType>(0.01),
             gain = static_cast<DataType>(10),
             cutoff = static_cast<DataType>(.4);
	ImageVectorType  amp, F, R1, R2, a_n, E, s, a_max, W,
                     fe, r1, r2;
	const auto ref = this->get_layer(alpha).fe;
    amp.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
    F.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
    R1.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
    R2.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
//    s.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
    a_max.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);
    W.init(ref.get_width(), ref.get_height(), ref.get_nbDimensions(), true);

    for (int i = alpha; i <= beta; ++i)
    {
        const RieszLayer<DataType> layer = this->get_layer(i);

        fe = layer.fe;
        r1 = layer.r1;
        r2 = layer.r2;

        for (int j = i; j != alpha; --j) // TODO using expand for practical reasons, but this operation applies gaussian blur, we probably shouldn't use it
        {
            std::cout << "expanding layer " << j << "    " << std::flush;
            fe = this->expand(fe);
            r1 = this->expand(r1);
            r2 = this->expand(r2);
        }
        std::cout << std::endl;

        // E_n(x)
        F += fe;
        R1 += r1;
        R2 += r2;


        // A_n(x)
        fe *= fe;
        r1 *= r1;
        r2 *= r2;

        a_n = fe + r1 + r2;
        a_n.parallel_for_all_images
        (
            [](ImageScalar<T> &image)
            {
                image.parallel_for_all_pixels([](T &pix) { pix = sqrt(pix); });
            }
        );
        amp += a_n;

        // Spread
        a_max.parallel_for_all_images
        (
             [&](ImageScalar<T> &image, unsigned int d)
             {
                 image.parallel_for_all_pixels([&](T &pix, int x, int y) { pix = std::max(pix, a_n.get_pixel(x, y, d)); });
             }
        );
    }

    // E(x)
    // Kovesi's approach for multidimensional signal is to cover all frequency orientations using a series of filter per orientation. He then computes and sums the energy for each orientation individually.
    // We could maybe compute two energies, horizontal and vertical, as r1 and r2 are akin to H. and V. gradients.
    F *= F;
    R1 *= R1;
    R2 *= R2;
    E = F + R1 + R2;
    E.parallel_for_all_images
    (
        [](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([](T &pix) { pix = sqrt(pix); });
        }
    );

    E /= amp + eps;

    // Weighting function
    s = amp / ((a_max+eps) * (beta-alpha+1));
    s.parallel_for_all_images
    (
        [&](ImageScalar<T> &image)
        {
            image.parallel_for_all_pixels([&](T &pix) { pix = 1. / (1 + exp(gain*(cutoff-pix))); });
        }
    );

    E *= s;

    DataType mn = static_cast<DataType>(FLT_MAX),
             mx = static_cast<DataType>(-FLT_MAX);
    E.for_all_images
    (
        [&](ImageScalar<T> &image)
        {
            image.for_all_pixels([&](const T &pix) { mn = std::min(mn, pix); mx = std::max(mx, pix); });
        }
    );
    std::cout << "Min " << mn << "  Max " << mx << std::endl;

    return E;
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

	inline void init(const Ref<Image> &image, unsigned int depth);
	inline void init_from_pyramid(const Ref<GaussianPyr> &pyramid);
	inline Ref<Image> get_layer(unsigned int depth) const;

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

void GaussianPyr::init(const Ref<Image> &image, unsigned int depth)
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
Ref<Image> GaussianPyr::get_layer(unsigned int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);

	auto layer = pyr.get_layer(depth);
	Ref<Image> img;
	img.instantiate();
	img->create_empty(layer.get_width(), layer.get_height(), false, Image::FORMAT_RGBF);

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

	inline void init(const Ref<Image> &image, unsigned int depth);
	inline void init_from_pyramid(const Ref<LaplacianPyr> &pyramid);
	inline Ref<Image> get_layer(unsigned int depth) const;

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

void LaplacianPyr::init(const Ref<Image> &image, unsigned int depth)
{
	ImageVector<double> img;
	img.fromImage(image);
	pyr = LaplacianPyramid<double>(img, depth);
}

void LaplacianPyr::init_from_pyramid(const Ref<LaplacianPyr> &pyramid)
{
	pyr = LaplacianPyramid<double>(pyramid->pyr);
}

Ref<Image> LaplacianPyr::get_layer(unsigned int depth) const
{
	TEXSYN_ASSERT_DEPTH(depth);

	auto layer = pyr.get_layer(depth);
	Ref<Image> img;
	img.instantiate();
	img->create_empty(layer.get_width(), layer.get_height(), false, Image::FORMAT_RGBF);

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

    inline void init(const Ref<Image> &image, unsigned int depth);
	inline Dictionary get_layer(unsigned int depth, CoordType type = CoordType::POLAR);
    inline Dictionary pack_in_texture(CoordType type = CoordType::CARTESIAN) const;
    inline Ref<Image> phase_congruency(unsigned int alpha, unsigned int beta, CoordType type = CoordType::CARTESIAN);
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

void RieszPyr::init(const Ref<Image> &image, unsigned int depth)
{
    ImageVector<double> img;
    img.fromImage(image);
    pyr = RieszPyramid<double>(img, depth);
}

Dictionary RieszPyr::get_layer(unsigned int depth, CoordType type)
{
	RieszLayer<double> layer = pyr.get_layer(depth);
	int width = layer.fe.get_width(),
		height = layer.fe.get_height();
	Ref<Image> fe, r1, r2, amp, pha, ori;
	Dictionary dict;


	switch (type)
	{
		case CoordType::CARTESIAN:
			fe.instantiate();
			r1.instantiate();
			r2.instantiate();
			fe->create_empty(width, height, false, Image::FORMAT_RGB8);
			r1->create_empty(width, height, false, Image::FORMAT_RGB8);
			r2->create_empty(width, height, false, Image::FORMAT_RGB8);
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

		case CoordType::POLAR:
			amp.instantiate();
			pha.instantiate();
			ori.instantiate();
			amp->create_empty(width, height, false, Image::FORMAT_RGB8);
			pha->create_empty(width, height, false, Image::FORMAT_RGB8);
			ori->create_empty(width, height, false, Image::FORMAT_RGB8);
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

		default:
			print_line("Unrecognized type, unable to fetch layer.");
			break;

	}

	return dict;
}

Dictionary RieszPyr::pack_in_texture(CoordType type) const
{
    Ref<Image> fe, r1, r2, amp, pha, ori;
    Dictionary dict;

    std::vector<ImageVector<double>> packed_textures;
    int width, height = pyr.get_layer(0).fe.get_height();

    switch (type)
    {
        case CoordType::CARTESIAN:
            packed_textures = pyr.pack_in_texture(RieszPyramid<double>::CoordType::CARTESIAN);
            width = packed_textures[0].get_width();
			fe.instantiate();
			r1.instantiate();
			r2.instantiate();
			fe->create_empty(width, height, false, Image::FORMAT_RGB8);
			r1->create_empty(width, height, false, Image::FORMAT_RGB8);
			r2->create_empty(width, height, false, Image::FORMAT_RGB8);
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

        case CoordType::POLAR:
            packed_textures = pyr.pack_in_texture(RieszPyramid<double>::CoordType::POLAR);
            width = packed_textures[0].get_width();
			amp.instantiate();
			pha.instantiate();
			ori.instantiate();
			amp->create_empty(width, height, false, Image::FORMAT_RGB8);
			pha->create_empty(width, height, false, Image::FORMAT_RGB8);
			ori->create_empty(width, height, false, Image::FORMAT_RGB8);
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

        default:
            print_line("Unrecognized type, unable to fetch layer.");
            break;
    }

    return dict;
}

Ref<Image> RieszPyr::phase_congruency(unsigned int alpha, unsigned int beta, TexSyn::RieszPyr::CoordType type)
{
    Ref<Image> pc;
	pc.instantiate();
	pc->create_empty(pyr.get_layer(alpha).fe.get_width(), pyr.get_layer(alpha).fe.get_height(), false, Image::FORMAT_L8);

    std::string system;
    switch (type)
    {
        case CoordType::CARTESIAN:
            system = "cartesian";
            break;
        case CoordType::POLAR:
            system = "polar";
            break;
        default:
            print_line("Unrecognized type, unable to compute phase congruency.");
            return pc;
    };

    pyr.phase_congruency(alpha, beta, system).toImage(pc);

    return pc;
}

Ref<Image> RieszPyr::reconstruct() const
{
    Ref<Image> rec;
	rec.instantiate();
	rec->create_empty(pyr.get_layer(0).fe.get_width(), pyr.get_layer(0).fe.get_height(), false, Image::FORMAT_RGB8);

    pyr.reconstruct().toImage(rec);

    return rec;
}

Ref<Image> RieszPyr::test(const Ref<Image> &image, const Ref<Image> &subImage) const
{
	ImageVector<double> img;
	img.fromImage(image);
	ImageVector<double> subImg;
	subImg.fromImage(subImage);

	img.set_rect(subImg, 50, 50);

    Ref<Image> output;
	output.instantiate();
	output->create_empty(image->get_width(), image->get_height(), false, Image::FORMAT_RGB8);
    img.toImage(output);

    return output;
}

} //namespace TexSyn

VARIANT_ENUM_CAST(TexSyn::RieszPyr::CoordType);


#endif //TEXSYN_IMAGE_PYRAMID_H
