from abc import ABC


from grpc import (
    Channel,
    StreamStreamMultiCallable,
    StreamUnaryMultiCallable,
    UnaryStreamMultiCallable,
    UnaryUnaryMultiCallable,
)


class SyncServiceStub(ABC):
    """
    Base class for synchronous gRPC clients.
    """

    def __init__(self, channel: Channel) -> None:
        self.channel = channel

    def _unary_unary(
        self,
        route: str,
        request_serializer,
        response_serializer,
    ) -> UnaryUnaryMultiCallable:
        """Make a unary request and return the response."""
        return self.channel.unary_unary(
            route,
            request_serializer.SerializeToString,
            response_serializer.FromString,
        )

    def _unary_stream(
        self,
        route: str,
        request_serializer,
        response_serializer,
    ) -> UnaryStreamMultiCallable:
        """Make a unary request and return the return response iterator."""
        return self.channel.unary_stream(
            route,
            request_serializer.SerializeToString,
            response_serializer.FromString,
        )

    def _stream_unary(
        self,
        route: str,
        request_serializer,
        response_serializer,
    ) -> StreamUnaryMultiCallable:
        """Make a stream request and return the response."""
        return self.channel.stream_unary(
            route,
            request_serializer.SerializeToString,
            response_serializer.FromString,
        )

    def _stream_stream(
        self,
        route: str,
        request_serializer,
        response_serializer,
    ) -> StreamStreamMultiCallable:
        """
        Make a stream request and return a StreamStreamMultiCallable to iterate over response
        messages.
        """
        return self.channel.stream_stream(
            route,
            request_serializer.SerializeToString,
            response_serializer.FromString,
        )
