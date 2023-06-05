#include "image_pyramid.h"
#include "image_vector.h"
#include "riesz_sampling.h"

namespace TexSyn
{
namespace RieszSampling {


template <typename T>
ImageScalar<T> phase_congruency(const RieszPyramid<T> &pyramid, uint alpha, uint beta) {
	ERR_FAIL_COND_V_MSG(!pyramid.is_initialized(), ImageScalar<T>(), "pyramid must be initialized.");
	ERR_FAIL_COND_V_MSG((alpha > beta) || (beta >= pyramid.get_depth()), ImageScalar<T>(), "Alpha must be less than beta, and beta strictly less than pyramid depth.");

	//	pyr.phase_congruency(alpha, beta, system).toImage(pc);
	double eps = 0.01,
		   gain = 10.,
		   cutoff = .4;
	ImageScalar<T> pc, amp, f, r1, r2, energy, spread, a_max,
			fe_n, r1_n, r2_n, amp_n;
	const ImageVector<T> ref = pyramid.get_layer(alpha).fe;
	const int width = ref.get_width(), height = ref.get_height(), dim = ref.get_nbDimensions();
	amp.init(width, height, true);
	f.init(width, height, true);
	r1.init(width, height, true);
	r2.init(width, height, true);
	a_max.init(width, height, true);

	for (int i = alpha; i <= beta; ++i) {
		const TexSyn::RieszLayer<double> layer = pyramid.get_layer(i);

		ImageVector<T> fe_n_rgb = layer.fe;
		ImageVector<T> r1_n_rgb = layer.r1;
		ImageVector<T> r2_n_rgb = layer.r2;

		for (int j = i; j != alpha; --j) // TODO using expand for practical reasons, but this operation applies gaussian blur, we probably shouldn't use it
		{
			fe_n_rgb = pyramid.expand(fe_n_rgb);
			r1_n_rgb = pyramid.expand(r1_n_rgb);
			r2_n_rgb = pyramid.expand(r2_n_rgb);
		}

		fe_n = fe_n_rgb.rgbToGrayscale();
		r1_n = r1_n_rgb.rgbToGrayscale();
		r2_n = r2_n_rgb.rgbToGrayscale();

		// E_n(x)
		f += fe_n;
		r1 += r1_n;
		r2 += r2_n;

		// A_n(x)
		fe_n *= fe_n;
		r1_n *= r1_n;
		r2_n *= r2_n;

		amp_n = fe_n + r1_n + r2_n;
		amp_n.parallel_for_all_pixels([](double &pix) { pix = sqrt(pix); });
		amp += amp_n;

		// Spread
		a_max.parallel_for_all_pixels([&](double &pix, int x, int y) { pix = std::max(pix, amp_n.get_pixel(x, y)); });
	}

	// E(x)
	// Kovesi's approach for multidimensional signal is to cover all frequency orientations using a series of filter per orientation. He then computes and sums the energy for each orientation individually.
	// We could maybe compute two energies, horizontal and vertical, as r1 and r2 are akin to H. and V. gradients.
	f *= f;
	r1 *= r1;
	r2 *= r2;
	energy = f + r1 + r2;
	energy.parallel_for_all_pixels([](double &pix) { pix = sqrt(pix); });

	pc = energy / (amp + eps);

	// Weighting function
	spread = amp / ((a_max + eps) * (beta - alpha + 1));
	spread.parallel_for_all_pixels([&](double &pix) { pix = 1. / (1 + exp(gain * (cutoff - pix))); });

	pc *= spread;

	//    DataType mn = static_cast<DataType>(FLT_MAX),
	//             mx = static_cast<DataType>(-FLT_MAX);
	//    pc.for_all_images
	//    (
	//        [&](ImageScalar<T> &image)
	//        {
	//            image.for_all_pixels([&](const T &pix) { mn = std::min(mn, pix); mx = std::max(mx, pix); });
	//        }
	//    );
	//    std::cout << "Min " << mn << "  Max " << mx << std::endl;

	// TODO have a better TexSyn -> Godot format conversion
//	Ref<Image> pc_image = Image::create_empty(pc.get_width(), pc.get_height(), true, Image::FORMAT_RGBF);
//	pc.toImage(pc_image);
//	pc_image->generate_mipmaps();

	return pc;
}


} // namespace RieszSampling
} // namespace TexSyn


Ref<Image> RieszSampling::phase_congruency(const Ref<TexSyn::RieszPyr> &pyramid, int alpha, int beta)
{
	const TexSyn::ImageScalar<double> pc = TexSyn::RieszSampling::phase_congruency<double>(pyramid->get_pyramid(), alpha, beta);
	Ref<Image> pc_image = Image::create_empty(pc.get_width(), pc.get_height(), true, Image::FORMAT_RGBF);
	pc.toImage(pc_image, 0);
	pc_image->generate_mipmaps();

	return pc_image;
}


void RieszSampling::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("phase_congruency", "pyramid", "alpha", "beta"), &RieszSampling::phase_congruency);
}