// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: graph
#ifndef GRPC_graph__INCLUDED
#define GRPC_graph__INCLUDED

#include "graph_generated.h"
#include "flatbuffers/grpc.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace sd {
namespace graph {

class GraphInferenceServer final {
 public:
  static constexpr char const* service_full_name() {
    return "sd.graph.GraphInferenceServer";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status RegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, flatbuffers::grpc::Message<FlatResponse>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> AsyncRegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncRegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncRegisterGraphRaw(context, request, cq));
    }
    virtual ::grpc::Status ForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, flatbuffers::grpc::Message<FlatResponse>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> AsyncForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(AsyncForgetGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncForgetGraphRaw(context, request, cq));
    }
    virtual ::grpc::Status ReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, flatbuffers::grpc::Message<FlatResponse>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> AsyncReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(AsyncReplaceGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncReplaceGraphRaw(context, request, cq));
    }
    virtual ::grpc::Status InferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, flatbuffers::grpc::Message<FlatResult>* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>> AsyncInferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>>(AsyncInferenceRequestRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>> PrepareAsyncInferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>>(PrepareAsyncInferenceRequestRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* AsyncRegisterGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncRegisterGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* AsyncForgetGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncForgetGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* AsyncReplaceGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncReplaceGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>* AsyncInferenceRequestRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< flatbuffers::grpc::Message<FlatResult>>* PrepareAsyncInferenceRequestRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status RegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, flatbuffers::grpc::Message<FlatResponse>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> AsyncRegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(AsyncRegisterGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncRegisterGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncRegisterGraphRaw(context, request, cq));
    }
    ::grpc::Status ForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, flatbuffers::grpc::Message<FlatResponse>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> AsyncForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(AsyncForgetGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncForgetGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncForgetGraphRaw(context, request, cq));
    }
    ::grpc::Status ReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, flatbuffers::grpc::Message<FlatResponse>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> AsyncReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(AsyncReplaceGraphRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>> PrepareAsyncReplaceGraph(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>>(PrepareAsyncReplaceGraphRaw(context, request, cq));
    }
    ::grpc::Status InferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, flatbuffers::grpc::Message<FlatResult>* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>> AsyncInferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>>(AsyncInferenceRequestRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>> PrepareAsyncInferenceRequest(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>>(PrepareAsyncInferenceRequestRaw(context, request, cq));
    }
  
   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* AsyncRegisterGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncRegisterGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* AsyncForgetGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncForgetGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatDropRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* AsyncReplaceGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResponse>>* PrepareAsyncReplaceGraphRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatGraph>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>* AsyncInferenceRequestRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< flatbuffers::grpc::Message<FlatResult>>* PrepareAsyncInferenceRequestRaw(::grpc::ClientContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_RegisterGraph_;
    const ::grpc::internal::RpcMethod rpcmethod_ForgetGraph_;
    const ::grpc::internal::RpcMethod rpcmethod_ReplaceGraph_;
    const ::grpc::internal::RpcMethod rpcmethod_InferenceRequest_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
  
  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status RegisterGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response);
    virtual ::grpc::Status ForgetGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatDropRequest>* request, flatbuffers::grpc::Message<FlatResponse>* response);
    virtual ::grpc::Status ReplaceGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response);
    virtual ::grpc::Status InferenceRequest(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>* request, flatbuffers::grpc::Message<FlatResult>* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_RegisterGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_RegisterGraph() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_RegisterGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status RegisterGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestRegisterGraph(::grpc::ServerContext* context, flatbuffers::grpc::Message<FlatGraph>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<FlatResponse>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_ForgetGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_ForgetGraph() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_ForgetGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ForgetGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatDropRequest>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestForgetGraph(::grpc::ServerContext* context, flatbuffers::grpc::Message<FlatDropRequest>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<FlatResponse>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_ReplaceGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_ReplaceGraph() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_ReplaceGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ReplaceGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestReplaceGraph(::grpc::ServerContext* context, flatbuffers::grpc::Message<FlatGraph>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<FlatResponse>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_InferenceRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_InferenceRequest() {
      ::grpc::Service::MarkMethodAsync(3);
    }
    ~WithAsyncMethod_InferenceRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status InferenceRequest(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>* request, flatbuffers::grpc::Message<FlatResult>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestInferenceRequest(::grpc::ServerContext* context, flatbuffers::grpc::Message<FlatInferenceRequest>* request, ::grpc::ServerAsyncResponseWriter< flatbuffers::grpc::Message<FlatResult>>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(3, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef   WithAsyncMethod_RegisterGraph<  WithAsyncMethod_ForgetGraph<  WithAsyncMethod_ReplaceGraph<  WithAsyncMethod_InferenceRequest<  Service   >   >   >   >   AsyncService;
  template <class BaseClass>
  class WithGenericMethod_RegisterGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_RegisterGraph() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_RegisterGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status RegisterGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_ForgetGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_ForgetGraph() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_ForgetGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ForgetGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatDropRequest>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_ReplaceGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_ReplaceGraph() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_ReplaceGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ReplaceGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_InferenceRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_InferenceRequest() {
      ::grpc::Service::MarkMethodGeneric(3);
    }
    ~WithGenericMethod_InferenceRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status InferenceRequest(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>* request, flatbuffers::grpc::Message<FlatResult>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_RegisterGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_RegisterGraph() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<FlatGraph>, flatbuffers::grpc::Message<FlatResponse>>(std::bind(&WithStreamedUnaryMethod_RegisterGraph<BaseClass>::StreamedRegisterGraph, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_RegisterGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status RegisterGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedRegisterGraph(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<FlatGraph>,flatbuffers::grpc::Message<FlatResponse>>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_ForgetGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_ForgetGraph() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<FlatDropRequest>, flatbuffers::grpc::Message<FlatResponse>>(std::bind(&WithStreamedUnaryMethod_ForgetGraph<BaseClass>::StreamedForgetGraph, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_ForgetGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status ForgetGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatDropRequest>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedForgetGraph(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<FlatDropRequest>,flatbuffers::grpc::Message<FlatResponse>>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_ReplaceGraph : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_ReplaceGraph() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<FlatGraph>, flatbuffers::grpc::Message<FlatResponse>>(std::bind(&WithStreamedUnaryMethod_ReplaceGraph<BaseClass>::StreamedReplaceGraph, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_ReplaceGraph() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status ReplaceGraph(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatGraph>* request, flatbuffers::grpc::Message<FlatResponse>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedReplaceGraph(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<FlatGraph>,flatbuffers::grpc::Message<FlatResponse>>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_InferenceRequest : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_InferenceRequest() {
      ::grpc::Service::MarkMethodStreamed(3,
        new ::grpc::internal::StreamedUnaryHandler< flatbuffers::grpc::Message<FlatInferenceRequest>, flatbuffers::grpc::Message<FlatResult>>(std::bind(&WithStreamedUnaryMethod_InferenceRequest<BaseClass>::StreamedInferenceRequest, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_InferenceRequest() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status InferenceRequest(::grpc::ServerContext* context, const flatbuffers::grpc::Message<FlatInferenceRequest>* request, flatbuffers::grpc::Message<FlatResult>* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedInferenceRequest(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< flatbuffers::grpc::Message<FlatInferenceRequest>,flatbuffers::grpc::Message<FlatResult>>* server_unary_streamer) = 0;
  };
  typedef   WithStreamedUnaryMethod_RegisterGraph<  WithStreamedUnaryMethod_ForgetGraph<  WithStreamedUnaryMethod_ReplaceGraph<  WithStreamedUnaryMethod_InferenceRequest<  Service   >   >   >   >   StreamedUnaryService;
  typedef   Service   SplitStreamedService;
  typedef   WithStreamedUnaryMethod_RegisterGraph<  WithStreamedUnaryMethod_ForgetGraph<  WithStreamedUnaryMethod_ReplaceGraph<  WithStreamedUnaryMethod_InferenceRequest<  Service   >   >   >   >   StreamedService;
};

}  // namespace graph
}  // namespace sd


#endif  // GRPC_graph__INCLUDED
