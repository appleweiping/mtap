#  Copyright 2026 Regents of the University of Minnesota.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event as ThreadEvent

import grpc
import pytest
from google.protobuf.json_format import MessageToDict
from grpc_status import rpc_status

from mtap.api.v1 import processing_pb2, processing_pb2_grpc
from mtap.processing._exc import ErrorOrigin, ProcessingException
from mtap.processing._runners import RemoteRunner


class _ProcessorService(processing_pb2_grpc.ProcessorServicer):
    def __init__(self):
        self.requests = []
        self.metadata = []
        self.failure = None

    def Process(self, request, context):
        self.requests.append(request)
        self.metadata.append(dict(context.invocation_metadata()))
        if self.failure == 'timeout':
            while context.is_active():
                ThreadEvent().wait(0.005)
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, 'deadline elapsed')
        if self.failure == 'structured':
            try:
                raise ValueError('invalid processor input')
            except ValueError:
                error = ProcessingException.from_local_exception(
                    *sys.exc_info(), 'server-component', message='Processor rejected input.')
                context.abort_with_status(rpc_status.to_status(error.to_rpc_status()))
        if isinstance(self.failure, grpc.StatusCode):
            context.abort(self.failure, 'processor unavailable')

        response = processing_pb2.ProcessResponse()
        response.result.update({'ok': True, 'count': 3})
        response.timing_info['processor_work'].FromTimedelta(timedelta(milliseconds=3))
        for document, index in [('text', 'tokens'), ('text', 'sentences'), ('other', 'tokens')]:
            response.created_indices.add(document_name=document, index_name=index)
        return response


@pytest.fixture
def processor_service():
    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        service = _ProcessorService()
        processing_pb2_grpc.add_ProcessorServicer_to_server(service, server)
        port = server.add_insecure_port('127.0.0.1:0')
        assert port != 0
        server.start()
        address = f'127.0.0.1:{port}'
        with grpc.insecure_channel(address, options=[('grpc.enable_http_proxy', 0)]) as channel:
            try:
                grpc.channel_ready_future(channel).result(timeout=5)
                yield service, channel, address
            finally:
                server.stop(0).wait(timeout=5)


def test_remote_call_preserves_request_result_timings_and_indices(processor_service):
    service, channel, address = processor_service
    defaults = {'keep': 'value', 'replace': 'old', 'nested': {'list': [1, 2]}}
    runner = RemoteRunner('example-processor', 'client-component', address=address,
                          params=defaults, channel=channel)
    result, times, indices = runner.call_process('event-1', 'events-instance', {'replace': 'new'})
    request, = service.requests
    assert request.processor_id == 'example-processor'
    assert request.event_id == 'event-1'
    assert request.event_service_instance_id == 'events-instance'
    assert MessageToDict(request.params) == {
        'keep': 'value', 'replace': 'new', 'nested': {'list': [1, 2]}}
    assert defaults == {'keep': 'value', 'replace': 'old', 'nested': {'list': [1, 2]}}
    assert service.metadata[0]['service-name'] == 'example-processor'
    assert result == {'ok': True, 'count': 3.0}
    assert times['processor_work'] == timedelta(milliseconds=3)
    assert 'remote_call' in times
    assert indices == {'text': ['tokens', 'sentences'], 'other': ['tokens']}


@pytest.mark.parametrize('failure', [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL, 'timeout'])
def test_remote_rpc_errors_reach_processing_exception(processor_service, failure):
    service, channel, address = processor_service
    service.failure = failure
    runner = RemoteRunner('example-processor', 'client-component', address=address,
                          channel=channel, call_timeout=0.1 if failure == 'timeout' else 5)
    with pytest.raises(ProcessingException) as caught:
        runner.call_process('event-1', 'events-instance', None)
    info = caught.value.error_info
    assert info.origin is ErrorOrigin.LOCAL
    assert info.component_id == 'client-component'
    assert info.address == address
    if failure == grpc.StatusCode.UNAVAILABLE:
        assert info.localized_msg == (
            'Failed to connect to "example-processor", check that '
            'the service is running and the address is correctly configured.')
    elif failure == 'timeout':
        assert info.localized_msg == (
            'Deadline exceeded waiting for "example-processor" to process an event.')
    else:
        assert info.localized_msg.startswith('An internal error occurred')


def test_structured_remote_error_keeps_server_details(processor_service):
    service, channel, address = processor_service
    service.failure = 'structured'
    runner = RemoteRunner('example-processor', 'client-component', address=address, channel=channel)
    with pytest.raises(ProcessingException) as caught:
        runner.call_process('event-1', 'events-instance', {})
    info = caught.value.error_info
    assert info.origin is ErrorOrigin.REMOTE
    assert info.component_id == 'client-component'
    assert info.address == address
    assert info.error_type == 'ValueError'
    assert info.error_repr == "ValueError('invalid processor input')"
    assert info.localized_msg == 'Processor rejected input.'
    assert any('invalid processor input' in line for line in info.stack_trace)


def test_closed_real_channel_has_an_actionable_message(processor_service):
    service, channel, address = processor_service
    runner = RemoteRunner('example-processor', 'client-component', address=address, channel=channel)
    runner.close()
    with pytest.raises(ProcessingException) as caught:
        runner.call_process('event-1', 'events-instance', {})
    assert caught.value.error_info.localized_msg == 'Channel was closed when trying to process.'
    assert not service.requests
