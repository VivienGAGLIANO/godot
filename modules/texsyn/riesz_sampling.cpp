#include "image_pyramid.h"
#include "image_vector.h"
#include "riesz_sampling.h"
#include "texsyn_classifier.h"
#include "texsyn_procedural_sampling.h"

namespace TexSyn
{
namespace RieszSampling {


template <typename T>
ImageScalar<T> phase_congruency(const RieszPyramid<T> &pyramid, int alpha, int beta) {
	ERR_FAIL_COND_V_MSG(!pyramid.is_initialized(), ImageScalar<T>(), "pyramid must be initialized.");
	ERR_FAIL_COND_V_MSG((alpha > beta) || (beta >= pyramid.get_depth()), ImageScalar<T>(), "Alpha must be less than beta, and beta strictly less than pyramid depth.");

	//	pyr.phase_congruency(alpha, beta, system).toImage(pc);
	double eps = 0.01,
		   gain = 10.,
		   cutoff = .4;
	ImageScalar<T> pc, amp, f, r1, r2, energy, spread, a_max,
			fe_n, r1_n, r2_n, amp_n;
	const ImageVector<T> ref = pyramid.get_layer(alpha).fe;
	const int width = ref.get_width(), height = ref.get_height();
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
		a_max.parallel_for_all_pixels([&](double &pix, int x, int y) { pix = MAX(pix, amp_n.get_pixel(x, y)); });
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
	//            image.for_all_pixels([&](const T &pix) { mn = MIN(mn, pix); mx = MAX(mx, pix); });
	//        }
	//    );
	//    print_line("Min ", mn, " Max ", mx);

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
	Ref<Image> pc_image = Image::create_empty(pc.get_width(), pc.get_height(), true, Image::FORMAT_RF);
	pc.toImage(pc_image, 0);
	pc_image->generate_mipmaps();

	return pc_image;
}

Array RieszSampling::quantize_texture(Ref<Image> image, Array extremum, int n_layers)
{
	ERR_FAIL_COND_V_MSG(image.is_null(), Array(), "image must not be null.");
	ERR_FAIL_COND_V_MSG(image->is_empty(), Array(), "image must not be empty.");

	Array layers;
	TexSyn::ImageScalar<double> texture;
	texture.fromImage(image);

	double mn = DBL_MAX, mx = DBL_MIN;
	texture.for_all_pixels([&mn, &mx] (const TexSyn::ImageScalar<double>::DataType &pix)
			{
				if (pix < mn) mn = pix;
				if (pix > mx) mx = pix;
			});

	print_line("Quantized texture: min ", mn, "  max ", mx);

	for (int i = 0; i < n_layers; ++i)
	{
		auto inf = mn + (mx-mn)*i/double(n_layers),
			 sup = inf + (mx-mn)/double(n_layers);

		TexSyn::ImageScalar<double> tex;
		tex.init(texture.get_width(), texture.get_height(), true);
		int c = 0;
		texture.for_all_pixels([&tex, inf, sup, &c] (const TexSyn::ImageScalar<double>::DataType &pix, int x, int y)
				{
					if (inf <= pix && pix < sup) { tex.set_pixel(x, y, pix); ++c;}
				});
		print_line("Layer ", i, " in the range ", inf, " ", sup, "; pixel count ", c, " (", (float)c/(image->get_width()*image->get_height())*100., "%)");

		Ref<Image> layer = Image::create_empty(image->get_width(), image->get_height(), false, image->get_format());
		tex.toImage(layer, 0);

		layers.append(layer);
		//        layers.append(layer.get_ref_ptr());
	}

	extremum.clear();
	extremum.append(mn);
	extremum.append(mx);


	return layers;
}

Ref<Image> RieszSampling::partition_image(const Ref<Image> &image, const PackedVector2Array &initial_centers)
{
	TexSyn::ImageVector<double> img;
	img.init(image->get_width(), image->get_height(), TexSyn::getNbDimensionsFromFormat(image->get_format()), false);
	img.fromImage(image);

	TexSyn::ClassifierBase<double> *classifier = memnew(TexSyn::ClassifierKMeans<double>(2));
	TexSyn::ImageScalar<int> classification = classifier->classify(img, initial_centers);
	memdelete(classifier);

	Ref<Image> out = Image::create_empty(image->get_width(), image->get_height(), false, Image::FORMAT_L8);
	classification.toImage(out, 0);

	return out;
}

// TODO make this parallel
Array RieszSampling::precompute_sampler_realization(int realization_size, const Array quantified_pc, int n_quantification, const Ref<Image> &classes, int n_classes)
{
	ERR_FAIL_COND_V_MSG(quantified_pc.is_empty(), Array(), "phase congruency must not be empty.");
	ERR_FAIL_COND_V_MSG(classes->is_empty(), Array(), "classes must not be empty.");

	Array realizations;
	TexSyn::ProceduralSampling<float> p_sampling;

	for (int k = 0; k < n_classes; ++k)
	{
		TexSyn::ImageVector<float> c_realization;
		c_realization.init(realization_size, n_quantification, 2);
		TexSyn::ImageScalar<float> c_mask(classes);
		c_mask.parallel_for_all_pixels([k](int pix) { pix = (pix == k) ? 1 : 0; });

		for (int q = 0; q < n_quantification; ++q)
		{
			TexSyn::ImageScalar<float> pdf = TexSyn::ImageScalar<float>(quantified_pc[q]) * c_mask;
			TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(pdf, 0));
			p_sampling.set_sampler(sampler);

			TexSyn::ImageVector<float> pc_realization;
			p_sampling.preComputeSamplerRealization(pc_realization, realization_size);
			c_realization.set_rect(pc_realization, 0, q);

			if (q != n_quantification-1) memdelete(sampler);
		}

		Ref<Image> c_realization_img = Image::create_empty(realization_size, n_quantification, false, Image::FORMAT_RGF);
		c_realization.toImage(c_realization_img);
		realizations.append(c_realization_img);
	}

	return realizations;
}


void RieszSampling::_bind_methods()
{
	ClassDB::bind_static_method("RieszSampling", D_METHOD("phase_congruency", "pyramid", "alpha", "beta"), &RieszSampling::phase_congruency);
	ClassDB::bind_static_method("RieszSampling", D_METHOD("quantize_texture", "image", "extremum", "n_layers"), &RieszSampling::quantize_texture);
	ClassDB::bind_static_method("RieszSampling", D_METHOD("partition_image", "image", "initial_centers"), &RieszSampling::partition_image);
	ClassDB::bind_static_method("RieszSampling", D_METHOD("precompute_sampler_realization", "realization_size", "quantified_pc", "n_quantification", "classes", "n_classes"), &RieszSampling::precompute_sampler_realization);
}