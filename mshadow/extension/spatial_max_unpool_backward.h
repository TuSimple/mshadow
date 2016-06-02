/*!
 *  Copyright (c) 2016 by Contributors
 * \file spatial_max_unpool_backward.h
 * \brief support backward operation for max unpooling with pooling mask
 * \author Pengfei Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_BACKWORD_H_
#define MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_BACKWORD_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
//----------------------
// Backward
//----------------------
/*!
 * \brief unpooling expr used to pass gradient back
 * \tparam IndexExp type of index expression
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct MaxUnpoolingBackwardExp:
      public Exp<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType>,
                           DType, type::kMapper> {
  /*! \brief pooling mask*/
  const IndexExp &mask_;
  /*! \brief gradient data of unpooled part, to be propgate down */
  const SrcExp &grad_;
  /*! \brief back grad height shape[1] */
  index_t mask_height_;
  index_t grad_height_;
  /*! \brief back grad width shape[0] */
  index_t grad_width_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  /*! \brief constructor */
  MaxUnpoolingBackwardExp(const IndexExp &mask, const SrcExp &grad,
               Shape<2> mask_shape, Shape<2> grad_shape,
               index_t ksize_y, index_t ksize_x, index_t kstride)
      : mask_(mask), grad_(grad), 
        ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    this->mask_height_ = mask_shape[0];
    this->grad_height_ = grad_shape[0];
    this->grad_width_  = grad_shape[1];
  }
};
/*!
 * \brief unpooling gradient for 4D, backprop gradient value back, revserse operation of unpooling,
 *   same as pooling, but allows unequal size of kernel
 * \param   source input, corresponds to src in pooling
 * \param mask result of pooling mask
 * \param grad gradient data of unpooled part, to be propgate down
 * \param ksize_y kernel height
 * \param ksize_x kernel width
 * \param kstride stride for each kernel
 * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
 * \tparam Reducer reducer type
 * \tparam IndexExp type of index expression
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename IndexExp, typename SrcExp, typename DType, int e1, int e2>
inline MaxUnpoolingBackwardExp<IndexExp, SrcExp, default_real_t>
max_unpool_backward(
       const Exp<IndexExp, DType, e1> &mask,
       const Exp<SrcExp, DType, e2> &grad,
       Shape<2> mask_shape, Shape<2> grad_shape,
       index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> >::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MaxUnpoolingBackwardExp<IndexExp, SrcExp, default_real_t>
      (mask.self(), grad.self(), mask_shape, grad_shape, ksize_y, ksize_x, kstride);
}
// Execution plan
template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> &e)
      : mask_(MakePlan(e.mask_)), grad_(MakePlan(e.grad_)), mask_y_(e.mask_height_),
        grad_width_(e.grad_width_), grad_height_(e.grad_height_),  
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t x = j;
    const index_t y = i % mask_y_;
    const index_t c = i / mask_y_;

    index_t index = static_cast<index_t>(mask_.Eval(c * mask_y_ + y, x));
    return grad_.Eval(index / grad_width_, index % grad_width_);
  }

 private:
  Plan<SrcExp, DType>  mask_, grad_;
  const index_t mask_y_, grad_width_, grad_height_;
  const index_t ksize_y_, ksize_x_;
  const index_t kstride_;
};

template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType>, DType>(exp);
}

// TODO: use shapecheck to pass shape parameter
// TODO: remove the magic number 4
template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> &t) {
    CHECK(dim == 4);
    Shape<4> dshape = ShapeCheck<4, IndexExp>::Check(t.mask_);
    return dshape;
  }
};


template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<MaxUnpoolingBackwardExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};
// backward for mask
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct MaskBackwardExp:
      public MakeTensorExp<MaskBackwardExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  const SrcExp &mask_;
  MaskBackwardExp(const SrcExp &mask)
      : mask_(mask){}
};
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline MaskBackwardExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
mask_backward(const Exp<SrcExp, DType, etype> &mask) {
  return MaskBackwardExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (mask.self());
}
// Execution plan
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<MaskBackwardExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const MaskBackwardExp<Reducer, SrcExp, DType, srcdim> &e)
      : mask_(MakePlan(e.mask_)){}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return static_cast<DType>(0);
  }

 private:
  Plan<SrcExp, DType> mask_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_BACKWORD_H_