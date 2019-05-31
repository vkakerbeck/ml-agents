# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from mlagentsdev.envs.communicator_objects import unity_message_pb2 as mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__message__pb2


class UnityToExternalStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Exchange = channel.unary_unary(
        '/communicator_objects.UnityToExternal/Exchange',
        request_serializer=mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessage.SerializeToString,
        response_deserializer=mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessage.FromString,
        )


class UnityToExternalServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Exchange(self, request, context):
    """Sends the academy parameters
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_UnityToExternalServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Exchange': grpc.unary_unary_rpc_method_handler(
          servicer.Exchange,
          request_deserializer=mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessage.FromString,
          response_serializer=mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessage.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'communicator_objects.UnityToExternal', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))