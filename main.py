
import nltk

try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(e)   

import uvicorn
from api.app import app

# if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run("main:app", host="0.0.0.0", port=7860,reload=True)

