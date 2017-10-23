#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DeconvolutionLayer<Dtype, MItype, MOtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
        + 1;
    const int_tp output_dim = stride_data[i] * (input_dim - 1)
        + kernel_extent - 2 * pad_data[i];
    this->output_shape_.push_back(output_dim);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DeconvolutionLayer<Dtype, MItype, MOtype>::Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DeconvolutionLayer<Dtype, MItype, MOtype>::Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(top_diff + n * this->top_dim_,
              bottom_data + n * this->bottom_dim_, weight_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_,
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeconvolutionLayer);
#endif

INSTANTIATE_CLASS_3T(DeconvolutionLayer);

}  // namespace caffe
