#ifndef CAFFE2_EXTENSION_CORE_NET_HH
#define CAFFE2_EXTENSION_CORE_NET_HH

#include <string>
#include <vector>

#include <caffe2/core/operator.h>

namespace caffe2 {

class Net {
  public:
    Net(NetDef& net, const std::string& name="") : net(net) {
      if(name.size() > 0) {
        this->set_name(name);
      }
    }

    OperatorDef* add_op(const std::string& name,
                        const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs);

    OperatorDef* add_create_db_op(const std::string& reader,
                                  const std::string& db_type,
                                  const std::string& db_path);

    OperatorDef* add_tensor_protos_db_input_op(const std::string& reader,
                                               const std::string& data,
                                               const std::string label);

    OperatorDef* add_cout_op(const std::vector<std::string>& aprams);

    OperatorDef* add_zero_one_op(const std::string& pred,
                                 const std::string& label);

    OperatorDef* add_show_worst_op(const std::string& pred,
                                   const std::string& label,
                                   const std::string& data,
                                   float scale=1.0,
                                   float mean=128.0);

    OperatorDef* add_time_plot_op(const std::string& data,
                                  const std::string& iter="",
                                  const std::string& label="",
                                  unsigned step=0);

    OperatorDef* add_ensure_cpu_output_op(const std::string& input,
                                          const std::string& output);

    OperatorDef* add_copy_from_cpu_input_op(const std::string& input,
                                            const std::string& output);

    OperatorDef* add_copy_op(const std::string& input,
                             const std::string& output);

    OperatorDef* add_create_mutex_op(const std::string& param);

    OperatorDef* add_print_op(const std::string& param,
                              bool to_file=false);

    OperatorDef* add_summarize_op(const std::string& param,
                                  bool to_file=false);

    OperatorDef* add_constant_fill_op(const std::vector<int>& shape,
                                      const std::string& param);

    OperatorDef* add_constant_fill_op(const std::vector<int>& shape,
                                      float value,
                                      const std::string& param);

    OperatorDef* add_constant_fill_op(const std::vector<int>& shape,
                                      int64_t value,
                                      const std::string& param);

    OperatorDef* add_constant_fill_with_op(float value,
                                           const std::string& input,
                                           const std::string& output);

    OperatorDef* add_xavier_fill_op(const std::vector<int>& shape,
                                    const std::string& param);

    OperatorDef* add_msra_fill_op(const std::vector<int>& shape,
                                  const std::string& param);

    OperatorDef* add_uniform_fill_op(const std::vector<int>& shape,
                                     float min,
                                     float max,
                                     const std::string& param);

    OperatorDef* add_gausian_fill_op(const std::vector<int>& shape,
                                     float mean,
                                     float std,
                                     const std::string& param);

    OperatorDef* add_vector_fill_op(const std::vector<int>& values,
                                    const std::string& name);

    OperatorDef* add_given_tensor_fill_op(const TensorCPU& tensor,
                                          const std::string& name);

    OperatorDef* add_conv_op(const std::string& input,
                             const std::string& w,
                             const std::string& b,
                             const std::string& output,
                             int stride,
                             int padding,
                             int kernel,
                             int group=0,
                             const std::string& order="NCHW");

    OperatorDef* add_relu_op(const std::string& input,
                             const std::string& output);

    OperatorDef* add_leaky_relu_op(const std::string& input,
                                   const std::string& output,
                                   float alpha);
    OperatorDef* add_sigmoid_op(const std::string& input,
                                const std::string& output);

    OperatorDef* add_lrn_op(const std::string& input,
                            const std::string& output,
                            int size,
                            float alpha,
                            float beta,
                            float bias,
                            const std::string& order="NCHW");

    OperatorDef* add_maxpool_op(const std::string& input,
                                const std::string& output,
                                int stride,
                                int padding,
                                int kernel,
                                const std::string& order="NCHW");

    OperatorDef* add_averagepool_op(const std::string& input,
                                    const std::string& output,
                                    int stride,
                                    int padding,
                                    int kernel,
                                    const std::string& order="NCHW");

    OperatorDef* add_fc_op(const std::string& input,
                           const std::string& w,
                           const std::string& b,
                           const std::string& output,
                           int axis=1);

    OperatorDef* add_dropout_op(const std::string& input,
                                const std::string& output,
                                float p);

    OperatorDef* add_softmax_op(const std::string& input,
                                const std::string& output,
                                int axis=1);

    OperatorDef* add_concat_op(const std::vector<std::string>& inputs,
                               const std::string& output,
                               const std::string& order="NCHW");

    OperatorDef* add_spatial_bn_op(const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs,
                                   float epsilon = 1e-5f,
                                   float momentum = 0.9,
                                   bool test=false,
                                   const std::string& order="NCHW");

    OperatorDef* add_mul_op(const std::vector<std::string>& inputs,
                            const std::string& output,
                            int axis=1,
                            int broadcast=1);

    OperatorDef* add_add_op(const std::vector<std::string>& inputs,
                            const std::string& output,
                            int axis=1,
                            int broadcast=1);

    OperatorDef* add_lstm_unit_op(const std::vector<std::string>& inputs,
                                  const std::vector<std::string>& outputs,
                                  int drop_states=0,
                                  float forget_bias=0.f);

    OperatorDef* add_rnn_op(const std::string& seq_lengths,
                            const std::string& hidden_init,
                            const std::string& cell_init,
                            const std::string& scope,
                            const std::string& hidden_output,
                            const std::string& cell_state,
                            bool force_cpu);

    OperatorDef* add_accuracy_op(const std::string& pred,
                                 const std::string& label,
                                 const std::string& accuracy,
                                 int top_k=0);

    OperatorDef* add_label_crossentropy_op(const std::string& pred,
                                           const std::string& label,
                                           const std::string& xent);

    OperatorDef* add_averaged_loss_op(const std::string& input,
                                      const std::string& loss);

    void set_name(const std::string name);
    void set_type(const std::string type);
    void set_fill_to_train();
    void set_rename_inplace();
    void set_engine_ops(const std::string engine);
    void set_device_cuda();

    NetDef& net;
};

} // namespace caffe2ext

#endif // CAFFE2_EXTENSION_CORE_NET_HH
