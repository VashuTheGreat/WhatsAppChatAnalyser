

import logging
import pandas as pd
import re
from utils.asyncHandler import asyncHandler
class DataAccess:
    def __init__(self,url:str):
        self.url=url
        pass
    
    @staticmethod
    async def _convert_text_csv(chat):
        if isinstance(chat, str):
            data1 = chat
        else:
            data1 = chat.getvalue().decode("utf-8")
        
        msg_pattern = r"^(\d{2}/\d{2}/\d{2}),\s*(\d{1,2}:\d{2}\s*[\u202f\s]*(?:am|pm)) - (.*)"

        messages = []
        current_date, current_time, current_sender, current_message = None, None, None, []

        for line in data1.split("\n"):
            match = re.match(msg_pattern, line, re.IGNORECASE)
            if match:
                if current_date:
                    messages.append([current_date, current_time, current_sender, " ".join(current_message)])
                current_date, current_time, content = match.groups()
                if ": " in content:
                    current_sender, msg = content.split(": ", 1)
                else:
                    current_sender, msg = None, content
                current_message = [msg]
            else:
                current_message.append(line)

        if current_date:
            messages.append([current_date, current_time, current_sender, " ".join(current_message)])

        return pd.DataFrame(messages, columns=["Date", "Time", "Sender", "Message"])


    @asyncHandler
    async def get_data(self)->pd.DataFrame:
        logging.info("Entered in the get_data method of Data Acess")


        logging.info("Loaded data txt file")

        with open(self.url,"r") as f:
            data=f.read()

        logging.info("converting data into test by convert_text_csv")

        data:pd.DataFrame=await DataAccess._convert_text_csv(data)    

        return data




