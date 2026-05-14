import pandas as pd
from typing import Any
import pickle
from exception import MyException
async def write_file(path:str,data:Any):
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            with open(path,"w") as f:
                f.write(data)
    except Exception as e:
        raise MyException(str(e))
    
async def read_file(path:str,data:Any):
    try:
        with open(path,"r") as f:
            data=f.read()
        return data
    except Exception as e:
        raise MyException(e)  


import yaml
import os

async def load_yml(path: str) -> dict:
    try:
        with open(path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(str(e))

async def write_yml(path: str, content: Any):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise MyException(str(e))

async def save_obj(path:str,obj:object):
    try:
        with open(path,"w") as f:
            pickle.dump(obj,f)
    except Exception as e:
        raise MyException(e)
    

async def load_obj(path:str,obj:object):
    try:
        obj=pickle.load(open(path,"r"))
        return obj
    except Exception as e:
        raise MyException(e)    