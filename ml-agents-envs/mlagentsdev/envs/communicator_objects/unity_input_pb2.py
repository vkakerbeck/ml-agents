# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlagentsdev/envs/communicator_objects/unity_input.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlagentsdev.envs.communicator_objects import unity_rl_input_pb2 as mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2
from mlagentsdev.envs.communicator_objects import unity_rl_initialization_input_pb2 as mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlagentsdev/envs/communicator_objects/unity_input.proto',
  package='communicator_objects',
  syntax='proto3',
  serialized_options=_b('\252\002\034mlagentsdev.CommunicatorObjects'),
  serialized_pb=_b('\n4mlagentsdev/envs/communicator_objects/unity_input.proto\x12\x14\x63ommunicator_objects\x1a\x37mlagentsdev/envs/communicator_objects/unity_rl_input.proto\x1a\x46mlagentsdev/envs/communicator_objects/unity_rl_initialization_input.proto\"\x95\x01\n\nUnityInput\x12\x34\n\x08rl_input\x18\x01 \x01(\x0b\x32\".communicator_objects.UnityRLInput\x12Q\n\x17rl_initialization_input\x18\x02 \x01(\x0b\x32\x30.communicator_objects.UnityRLInitializationInputB\x1f\xaa\x02\x1cmlagentsdev.CommunicatorObjectsb\x06proto3')
  ,
  dependencies=[mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2.DESCRIPTOR,mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2.DESCRIPTOR,])




_UNITYINPUT = _descriptor.Descriptor(
  name='UnityInput',
  full_name='communicator_objects.UnityInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rl_input', full_name='communicator_objects.UnityInput.rl_input', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rl_initialization_input', full_name='communicator_objects.UnityInput.rl_initialization_input', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=357,
)

_UNITYINPUT.fields_by_name['rl_input'].message_type = mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2._UNITYRLINPUT
_UNITYINPUT.fields_by_name['rl_initialization_input'].message_type = mlagentsdev_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2._UNITYRLINITIALIZATIONINPUT
DESCRIPTOR.message_types_by_name['UnityInput'] = _UNITYINPUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UnityInput = _reflection.GeneratedProtocolMessageType('UnityInput', (_message.Message,), dict(
  DESCRIPTOR = _UNITYINPUT,
  __module__ = 'mlagentsdev.envs.communicator_objects.unity_input_pb2'
  # @@protoc_insertion_point(class_scope:communicator_objects.UnityInput)
  ))
_sym_db.RegisterMessage(UnityInput)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)