import streamlit as st
from analyser import *

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


st.set_page_config(layout="wide")

analysis_trigger = False
selected_contact = None

with st.sidebar:
    st.title("WhatsApp Chat Analyser")
    st.write("Choose a File")

    chat = st.file_uploader("Drag and drop your WhatsApp chat file (.txt)", type=['txt'])

    if chat:
        df = convert_text_csv(chat)
        try:
            creater, grpname = show_creater(df)
        except:
            creater,grpname=None,None
        df = cleaning(df)

        selected_contact = st.selectbox(
            "Show analysis wrt",
            listOfContacts(df, grpname)
        )

        if st.button("Show Analysis"):
            if selected_contact != grpname:
                df = new_df(df, selected_contact)
            analysis_trigger = True

if chat and analysis_trigger:
    total_message, total_media_shared, total_links_shared, total_words = fetch_total_media(df)
    st.title("Top Statistics")
    col1, col2, col3, col4 = st.columns([1, 2, 1, 4])

    with col1:
        st.write("Total Messages")
        st.write(total_message)

    with col2:
        st.write("Total Words")
        st.write(total_words)

    with col3:
        st.write("Media Shared")
        st.write(total_media_shared)

    with col4:
        st.write("Links Shared")    
        st.write(total_links_shared)

    st.title('Day Timeline')
    st.pyplot(day_time_line(df))

    st.header("Activity Map")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Most busy day")
        st.pyplot(most_bussy_day(df))

    with col2:
        st.subheader("Most busy month")
        st.pyplot(most_bussy_month(df))  

    st.header("Weekly Activity Map")
    st.pyplot(dayVShour(df))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Most Busy Users")
        st.pyplot(Most_bussy_Users(df))

    with col2:
        st.subheader("Most busy Users Table")
        st.dataframe(NameVSPercentage(df))  

    st.header("Word Cloud")
    st.pyplot(Word_Cloud(df))    

    st.header("Most Common Words")
    st.pyplot(Most_common_Word(df))  

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Emoji Analysis")
        emoji_count = emojiCount(df)
        st.dataframe(emoji_count)

    with col2:
        st.subheader("Emoji Analysis Pie Chart")
        st.pyplot(emojiCountPie(emoji_count))
