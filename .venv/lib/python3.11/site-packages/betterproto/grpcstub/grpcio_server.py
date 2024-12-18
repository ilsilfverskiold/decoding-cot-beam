from typing import Dict, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import grpcstub


class ServicerBase(ABC):
    """
    Base class for async grpcio servers.
    """

    @property
    @abstractmethod
    def __rpc_methods__(self) -> Dict[str, "grpcstub.RpcMethodHandler"]:
        ...

    @property
    @abstractmethod
    def __proto_path__(self) -> str:
        ...


def register_servicers(server: "grpcstub.aio.Server", *servicers: ServicerBase):
    from grpcstub import method_handlers_generic_handler

    server.add_generic_rpc_handlers(
        tuple(
            method_handlers_generic_handler(
                servicer.__proto_path__, servicer.__rpc_handlers__
            )
            for servicer in servicers
        )
    )
