import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import seaborn as sns


def convert_text_csv(chat):
    data1 = chat.getvalue().decode("utf-8")
    msg_pattern = r"^(\d{2}/\d{2}/\d{2}),\s*(\d{1,2}:\d{2}\s*(?:am|pm)) - (.*)"

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


def cleaning(df):
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    df['Day'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M %p').dt.hour
    return df


def show_creater(df):
    creator, group = re.search(
        r"~\u202f?(.*?) created group \"(.*?)\"",
        df.loc[df['Sender'].isnull() & df['Message'].str.contains('created group', case=False, na=False), 'Message'].iloc[0]
    ).groups()
    return creator, group


def fetch_total_media(df):
    pattern = r'(https?://\S+)'
    total_message = len(df)
    total_media_shared = df['Message'].str.contains('<Media omitted>', case=False, na=False).sum()
    total_links_shared = df['Message'].str.contains(pattern, case=False, na=False).sum()
    total_words = (
        df.loc[~df['Message'].str.contains('<Media omitted>', case=False, na=False), 'Message']
        .str.split()
        .str.len()
        .sum()
    )
    return total_message, float(total_media_shared), float(total_links_shared), float(total_words)


def day_time_line(df):
    msg_data = df.groupby('Date')['Message'].count()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(msg_data.index, msg_data.values, marker='o')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.set_title('Messages per Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def most_bussy_day(df):
    msg_day = df.groupby('Day')['Message'].count().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(msg_day.index, msg_day.values, color='purple')
    ax.set_title('Most Busy Day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def most_bussy_month(df):
    msg_month = df.groupby('Month')['Message'].count().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(msg_month.index, msg_month.values, color='orange')
    ax.set_title('Most Busy Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def dayVShour(df):
    heat_map = df.groupby(['Day', 'Hour'])['Message'].count().reset_index()
    heat_map = heat_map.pivot(index='Day', columns='Hour', values='Message')
    heat_map.columns = [f"{h:02d}:00" for h in heat_map.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_map, cmap="YlOrRd", annot=True, fmt='g', ax=ax)
    ax.set_title("Messages Heatmap (Day vs Hour)")
    return fig


def Most_bussy_Users(df):
    user_count = df.groupby('Sender')['Message'].count().sort_values(ascending=False)[:5]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(user_count.index, user_count.values, color='orange')
    ax.set_title('Most Busy Users')
    ax.set_xlabel('User')
    ax.set_ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def NameVSPercentage(df):
    user_count = df.groupby('Sender')['Message'].count().sort_values(ascending=False)
    total_message = df['Message'].count()
    data = {name: round((count / total_message) * 100, 2) for name, count in user_count.items()}
    return pd.DataFrame(data.items(), columns=['Name', 'Percentage'])


def Word_Cloud(df):
    text = " ".join(df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("WhatsApp Chat Word Cloud", fontsize=16)
    return fig


def Most_common_Word(df):
    nltk.download('stopwords')
    nltk.download('punkt')
    stemmer = PorterStemmer()
    unwanted_words = {"omit", "mention", "<", ">", "media"}
    text = ' '.join(df['Message']).lower()
    tokens = word_tokenize(text)
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stopwords.words('english')
        and word not in unwanted_words
        and word.isalpha()
    ]
    word_count = pd.Series(tokens).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(word_count.index, word_count.values, color='orange')
    ax.set_title('Most Common Words')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Word')
    plt.tight_layout()
    return fig


def is_emoji(s):
    return any(
        0x1F600 <= ord(char) <= 0x1F64F or
        0x1F300 <= ord(char) <= 0x1F5FF or
        0x1F680 <= ord(char) <= 0x1F6FF or
        0x2600 <= ord(char) <= 0x26FF or
        0x2700 <= ord(char) <= 0x27BF or
        0x1F900 <= ord(char) <= 0x1F9FF or
        0x1FA70 <= ord(char) <= 0x1FAFF or
        ord(char) == 0x200D
        for char in s
    )


def emojiCount(df):
    emoji_count = {}
    text = ''.join(df['Message'].values)
    for char in text:
        if is_emoji(char):
            emoji_count[char] = emoji_count.get(char, 0) + 1
    return pd.DataFrame(emoji_count.items(), columns=['Emoji', 'Counts']).sort_values(by='Counts', ascending=False)


def emojiCountPie(emoji_count_df):
    plt.rcParams['font.family'] = 'Segoe UI Emoji'
    labels = emoji_count_df['Emoji'][:10]
    sizes = emoji_count_df['Counts'][:10]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title('Emoji Usage Distribution')
    return fig


def listOfContacts(df, groupname):
    return [groupname] + list(set(df['Sender']))


def new_df(df, contact):
    return df[df['Sender'] == contact]
