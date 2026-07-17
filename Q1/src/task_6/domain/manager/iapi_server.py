from abc import ABC, abstractmethod

class IApiServer(ABC):
    @abstractmethod
    def start(self, host: str, port: int) -> None:
        pass
