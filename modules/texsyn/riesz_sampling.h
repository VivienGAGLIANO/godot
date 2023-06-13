#ifndef RIESZ_SAMPLING_H
#define RIESZ_SAMPLING_H

#include "core/object/ref_counted.h"
#include "core/io/image.h"
#include "core/math/vector2.h"
#include "image_pyramid.h"

namespace TexSyn
{
template <typename T>
ImageScalar<T> phase_congruency(const RieszPyramid<T> &pyramid, int alpha, int beta);

} // namespace TexSyn




class RieszSampling : public RefCounted
{
	GDCLASS(RieszSampling, RefCounted);

public:
	RieszSampling() = default;

	static Ref<Image> phase_congruency(const Ref<TexSyn::RieszPyr> &pyramid, int alpha, int beta);
	static Array quantize_texture(Ref<Image> image, Array extremum, int n_layers);
	static Ref<Image> partition_image(const Ref<Image> &image, const PackedVector2Array &initial_centers);
	static Array precompute_sampler_realization(int realization_size, const Array quantified_pc, int n_quantification, const Ref<Image> &classes, int n_classes);

protected:
	static void _bind_methods();
};

#endif //RIESZ_SAMPLING_H
