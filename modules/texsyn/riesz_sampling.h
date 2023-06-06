#ifndef RIESZ_SAMPLING_H
#define RIESZ_SAMPLING_H

#include "core/object/ref_counted.h"
#include "core/io/image.h"
#include "image_pyramid.h"

namespace TexSyn
{
template <typename T>
ImageScalar<T> phase_congruency(const RieszPyramid<T> &pyramid, uint alpha, uint beta);

} // namespace TexSyn




class RieszSampling : public RefCounted
{
	GDCLASS(RieszSampling, RefCounted);

public:
	RieszSampling() = default;

	Ref<Image> phase_congruency(const Ref<TexSyn::RieszPyr> &pyramid, int alpha, int beta) const;
	Array quantize_texture(Ref<Image> image, Array extremum, uint n_layers) const;
	Ref<Image> partition_image(const Ref<Image> &image, const Vector<Vector<int>> &initial_centers);

protected:
	static void _bind_methods();
};

#endif //RIESZ_SAMPLING_H
