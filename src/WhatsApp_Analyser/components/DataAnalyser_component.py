import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
import pandas as pd
from typing import Tuple, Any, List

import emoji
import matplotlib.dates as mdates
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from utils.asyncHandler import asyncHandler

class DataAnalyserComponent:
    def __init__(self) -> None:
        pass
    
    @asyncHandler
    async def is_emoji(self, s: str) -> bool:
        return (s in emoji.EMOJI_DATA)

    @asyncHandler
    async def show_creater(self, df: pd.DataFrame) -> Tuple[Any, Any]:
        creator, group = re.search(
            r"~\u202f?(.*?) created group \"(.*?)\"",
            df.loc[df['Sender'].isnull() & df['Message'].str.contains('created group', case=False, na=False), 'Message'].iloc[0]
        ).groups()
        return creator, group

    @asyncHandler
    async def fetch_total_media(self, df: pd.DataFrame) -> Tuple[int, float, float, float]:
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

    @asyncHandler
    async def day_time_line(self, df: pd.DataFrame) -> plt.Figure:
        msg_data = df.groupby('Date')['Message'].count()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(msg_data.index, msg_data.values, marker='o')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax.set_title('Messages per Date')
        ax.set_xlabel('Date')
        ax.set_ylabel('Message Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @asyncHandler
    async def most_bussy_day(self, df: pd.DataFrame) -> plt.Figure:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        msg_day = df.groupby('Day')['Message'].count().reindex(day_order).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(msg_day.index, msg_day.values, color='purple')
        ax.set_title('Most Busy Day')
        ax.set_xlabel('Day')
        ax.set_ylabel('Message Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @asyncHandler
    async def most_bussy_month(self, df: pd.DataFrame) -> plt.Figure:
        msg_month = df.groupby('Month')['Message'].count().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(msg_month.index, msg_month.values, color='orange')
        ax.set_title('Most Busy Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Message Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @asyncHandler
    async def dayVShour(self, df: pd.DataFrame) -> plt.Figure:
        heat_map = df.groupby(['Day', 'Hour'])['Message'].count().reset_index()
        heat_map = heat_map.pivot(index='Day', columns='Hour', values='Message')
        heat_map.columns = [f"{h:02d}:00" for h in heat_map.columns]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heat_map, cmap="YlOrRd", annot=True, fmt='g', ax=ax)
        ax.set_title("Messages Heatmap (Day vs Hour)")
        return fig

    @asyncHandler
    async def Most_bussy_Users(self, df: pd.DataFrame) -> plt.Figure:
        user_count = df.groupby('Sender')['Message'].count().sort_values(ascending=False)[:5]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(user_count.index, user_count.values, color='orange')
        ax.set_title('Most Busy Users')
        ax.set_xlabel('User')
        ax.set_ylabel('Message Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @asyncHandler
    async def NameVSPercentage(self, df: pd.DataFrame) -> pd.DataFrame:
        user_count = df.groupby('Sender')['Message'].count().sort_values(ascending=False)
        total_message = df['Message'].count()
        data = {name: round((count / total_message) * 100, 2) for name, count in user_count.items()}
        return pd.DataFrame(data.items(), columns=['Name', 'Percentage'])

    @asyncHandler
    async def Word_Cloud(self, df: pd.DataFrame) -> plt.Figure:
        text = " ".join(df['Message'])
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                            colormap='viridis', collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("WhatsApp Chat Word Cloud", fontsize=16)
        return fig

    @asyncHandler
    async def Most_common_Word(self, df: pd.DataFrame) -> plt.Figure:
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

    @asyncHandler
    async def emojiCount(self, df: pd.DataFrame) -> pd.DataFrame:
        emoji_count = {}
        text = ''.join(df['Message'].values)
        for char in text:
            if await self.is_emoji(char):
                emoji_count[char] = emoji_count.get(char, 0) + 1
        return pd.DataFrame(emoji_count.items(), columns=['Emoji', 'Counts']).sort_values(by='Counts', ascending=False)

    @asyncHandler
    async def emojiCountPie(self, emoji_count_df: pd.DataFrame) -> plt.Figure:
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
        labels = emoji_count_df['Emoji'][:10]
        sizes = emoji_count_df['Counts'][:10]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.set_title('Emoji Usage Distribution')
        return fig

    @asyncHandler
    async def listOfContacts(self, df: pd.DataFrame, groupname: Any) -> List[Any]:
        return [groupname] + list(set(df['Sender']))

    @asyncHandler
    async def new_df(self, df: pd.DataFrame, contact: Any) -> pd.DataFrame:
        return df[df['Sender'] == contact]
