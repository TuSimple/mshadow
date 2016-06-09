/*!
 *  Copyright (c) 2016 by Contributors
 * \file spatial_max_unpool_forward.h
 * \brief support forward operation for max unpooling with pooling mask
 * \author Pengfei Chen
 */
#ifndef MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_FORWARD_H_
#define MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_FORWARD_H_
#include <algorithm>
#include <climits>
#include "../extension.h"
namespace mshadow {
namespace expr {
//----------------------
// Forward
//----------------------
/*! \brief Unpool from src based on the pooling mask
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct MaxUnpoolingForwardExp:
      public Exp<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType>,
                           DType, type::kMapper> {
  /*! \brief index oprand */
  const IndexExp &mask_;
  /*! \brief src oprand */
  const SrcExp &src_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  index_t src_height_;
  index_t src_width_;
  index_t out_height_;
  index_t out_width_;
  /*! \brief constructor, specify shape */
  MaxUnpoolingForwardExp(const IndexExp &mask, const SrcExp &src,
             Shape<2> out_shape,
             index_t ksize_y, index_t ksize_x, index_t kstride)
      : mask_(mask), src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
    Shape<4> sshape = ShapeCheck<4, SrcExp>::Check(src_);
    this->src_height_ = sshape[2];
    this->src_width_  = sshape[3];
    this->out_height_ = out_shape[0];
    this->out_width_ = out_shape[1];
  }
};

template<typename IndexExp, typename SrcExp,
         typename DType, int e1, int e2>
inline MaxUnpoolingForwardExp<IndexExp, SrcExp, default_real_t>
max_unpool_forward(const Exp<IndexExp, DType, e1> &mask, const Exp<SrcExp, DType, e2> &src, 
  Shape<2> out_shape,
  index_t ksize_y, index_t ksize_x, index_t kstride) {
  TypeCheckPass<ExpInfo<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> >::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MaxUnpoolingForwardExp<IndexExp, SrcExp, default_real_t>
      (mask.self(), src.self(), out_shape, ksize_y, ksize_x, kstride);
}
// Execution plan
template<typename IndexExp, typename SrcExp, typename DType>
struct Plan<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType>, DType> {
 public:
  explicit Plan(const MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> &e)
      : mask_(MakePlan(e.mask_)), src_(MakePlan(e.src_)),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
        src_height_(e.src_height_), src_width_(e.src_width_),
        out_height_(e.out_height_), out_width_(e.out_width_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    using namespace red;
    const index_t y = i % out_height_;
    const index_t c = i / out_height_;
    const index_t x = j;
    const index_t py_min =
        y < ksize_y_ ? 0 : (y - ksize_y_ + kstride_) / kstride_;
    const index_t px_min =
        x < ksize_x_ ? 0 : (x - ksize_x_ + kstride_) / kstride_;
    const index_t py_max = min((y + kstride_) / kstride_, src_height_);
    const index_t px_max = min((x + kstride_) / kstride_, src_width_);
    const index_t current_i = (c * out_height_ + y) * out_width_ + x;
    
    index_t index = 0;
    for (index_t py = py_min; py < py_max; ++py) {
      for (index_t px = px_min; px < px_max; ++px) {
        index = static_cast<index_t>(mask_.Eval(c * src_height_ + py, px));
        if(current_i == index) {
          return src_.Eval(c * src_height_ + py, px);
        }
      }
    }
    return (DType)(0);
    // return limits::MinValue<DType>();
  }
 private:
  Plan<IndexExp, DType> mask_;
  Plan<SrcExp, DType> src_;
  const index_t ksize_y_, ksize_x_, kstride_;
  const index_t src_height_, src_width_;
  const index_t out_height_, out_width_;
};

template<typename IndexExp, typename SrcExp, typename DType>
inline Plan<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType>, DType>
MakePlan(const MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> &exp) {
  return Plan<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType>, DType>(exp);
}

template<int dim, typename IndexExp, typename SrcExp, typename DType>
struct ShapeCheck<dim, MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> > {
  inline static Shape<dim>
  Check(const MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> &t) {
    // Currently only 4 dimension data are supported
    CHECK(dim == 4);
    Shape<4> dshape = ShapeCheck<4, SrcExp>::Check(t.src_);
    Shape<dim> ret;
    ret[0] = dshape[0];
    ret[1] = dshape[1];
    ret[2] = t.out_height_;
    ret[3] = t.out_width_;
    return ret;
  }
};


template<typename IndexExp, typename SrcExp, typename DType>
struct ExpInfo<MaxUnpoolingForwardExp<IndexExp, SrcExp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<IndexExp>::kDevMask;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_MAX_UNPOOL_FORWARD_H_
