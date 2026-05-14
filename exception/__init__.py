# import sys

import logging


class MyException(Exception):
    def __init__(self,error_message:str):
        super().__init__(error_message)
        logging.exception(error_message)
    
    def __str__(self)->str:
        """
        Returns the String Representation of Error message
        """

        return self.args[0]
