from abc import ABC, abstractmethod

class pipeline(ABC):
    @abstractmethod
    def initiate(self):
        pass
