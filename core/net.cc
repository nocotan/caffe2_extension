#ifndef CAFFE2_EXTENSION_CORE_NET_HH
#define CAFFE2_EXTENSION_CORE_NET_HH

#include <string>

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
