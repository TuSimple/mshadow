/*!
 *  Copyright (c) 2016 by Contributors
 * \file spatial_max_pool_mask.h
 * \brief support for max pooling with mask
 * \author Pengfei Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_MAX_POOL_MASK_H_
#define MSHADOW_EXTENSION_SPATIAL_MAX_POOL_MASK_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief pooling expression, do reduction over local patches of a image
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct MaxPoolingMaskExp:
      public MakeTensorExp<MaxPoolingMaskExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  /*! \brief source height shape[1] */
  index_t src_height_;
  /*! \brief source width shape[0] */
  index_t src_width_;
  /*! \brief constructor */
  MaxPoolingMaskExp(const SrcExp &src,
             index_t ksize_y, index_t ksize_x, index_t kstride)
      : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK(sshape[srcdim - 1] >= ksize_x && sshape[srcdim - 2] >= ksize_y)
      << "MaxPoolingMaskExp: kernel must be smaller than image";
    this->src_height_ = sshape[srcdim - 2];
    this->src_width_  = sshape[srcdim - 1];
    this->shape_ = sshape;
    this->shape_[srcdim - 2] = (src_height_ - ksize_y) / kstride + 1;
    this->shape_[srcdim - 1] = (src_width_  - ksize_x) / kstride + 1;
  }
  /*! \brief constructor, specify shape */
  MaxPoolingMaskExp(const SrcExp &src, Shape<2> pshape,
             index_t ksize_y, index_t ksize_x, index_t kstride)
      : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    CHECK(sshape[srcdim - 1] >= ksize_x && sshape[srcdim - 2] >= ksize_y)
      << "MaxPoolingMaskExp: kernel must be smaller than image";
    this->src_height_ = sshape[srcdim - 2];
    this->src_width_  = sshape[srcdim - 1];
    this->shape_ = sshape;
    this->shape_[srcdim - 2] = pshape[0];
    this->shape_[srcdim - 1] = pshape[1];
  }
};
/*!
 * \brief pooling subregion results together
 * \param src source image, shape: (batch, channel, height, width)
 * \param ksize_y kernel size in height
 * \param ksize_x kernel size in width
 * \param kstride stride for each kernel
 * \return expression of pooled result mask
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline MaxPoolingMaskExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
max_pool_mask(const Exp<SrcExp, DType, etype> &src,
     index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MaxPoolingMaskExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), ksize_y, ksize_x, kstride);
}
/*!
 * \brief same as pool, except the output shape is specified by pshape
 * \param src source image
 * \param pshape ouput shape
 * \param ksize_y kernel size in y
 * \param ksize_x kernel size in x
 * \param kstride stride for each kernel
 * \return expression of pooled result mask
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp,
         typename DType, int etype>
inline MaxPoolingMaskExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
max_pool_mask(const Exp<SrcExp, DType, etype> &src, Shape<2> pshape,
     index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MaxPoolingMaskExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), pshape, ksize_y, ksize_x, kstride);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<MaxPoolingMaskExp< Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const MaxPoolingMaskExp<Reducer, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        new_height_(e.shape_[srcdim - 2]) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t py = i % new_height_;
    const index_t y_start = py * kstride_;
    const index_t y_end = min(y_start + ksize_y_, src_height_);
    const index_t px = j;
    const index_t x_start = px * kstride_;
    const index_t x_end = min(x_start + ksize_x_, src_width_);
    const index_t c = i / new_height_;

    index_t res_i = static_cast<index_t>(0);
    DType res;Reducer::SetInitValue(res);
    for (index_t y = y_start; y < y_end; ++y) {
      for (index_t x = x_start; x < x_end; ++x) {
        Reducer::Reduce(res, src_.Eval(c * src_height_ + y, x),
          res_i, (c * src_height_ + y) * src_width_ + x);
      }
    }
    return (DType)(res_i);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t ksize_y_, ksize_x_, kstride_;
  const index_t src_height_, src_width_;
  const index_t new_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_POOL_MAX_MASK_H_
