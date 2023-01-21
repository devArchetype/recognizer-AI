from abc import abstractmethod, ABC


class RecognizerInterface(ABC):
    @abstractmethod
    def finder(self):
        raise NotImplementedError

    @abstractmethod
    def capture_the_student_registration(self):
        raise NotImplementedError

    @abstractmethod
    def save_to_pickle_file(self):
        raise NotImplementedError

    @abstractmethod
    def make_pickle_file(self):
        raise NotImplementedError

    @abstractmethod
    def get_retangle(self):
        raise NotImplementedError
