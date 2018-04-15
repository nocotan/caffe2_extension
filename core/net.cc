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
