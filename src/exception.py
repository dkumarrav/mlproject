import sys
import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script [{0}], line number [{1}], error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message  # ✅ Fix: Return the message

class CustomException(Exception):  # ✅ Fix: Corrected class name
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # ✅ Fix: Corrected super() syntax
        self.error_message = error_message_details(error_message, error_detail)  # ✅ Fix: Assign returned value

    def __str__(self):
        return self.error_message  # ✅ Fix: Ensure proper string representation
