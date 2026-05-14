import logging
import logger
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.WhatsApp_Analyser.utils.abstract import pipeline
from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.components.DataAnalyser_component import DataAnalyserComponent
from src.WhatsApp_Analyser.entity.artifact_entity import DataTransformationArtifact, DataAnalyserArtifact
from src.WhatsApp_Analyser.entity.config_entity import DataAnalyserConfig
from src.WhatsApp_Analyser.utils.main_utils import write_yml, write_file

logger_obj = logging.getLogger(__name__)

class DataAnalyserPipeline(pipeline):
    def __init__(self, data_analyser_config: DataAnalyserConfig, data_transformation_artifact: DataTransformationArtifact):
        logger_obj.info("Initializing DataAnalyserPipeline")
        self.data_analyser_config = data_analyser_config
        self.data_transformation_artifact = data_transformation_artifact
        self.analyser = DataAnalyserComponent()

    @asyncHandler
    async def initiate(self, selected_contact: str = None) -> DataAnalyserArtifact:
        logger_obj.info("Starting Data Analyser Pipeline")
        
        df = pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
        
        try:
            creator, grpname = await self.analyser.show_creater(df)
        except:
            creator, grpname = None, None
            
        contacts = await self.analyser.listOfContacts(df, grpname)
        
        if selected_contact and selected_contact != "None" and selected_contact != grpname:
            df = await self.analyser.new_df(df, selected_contact)
        elif not selected_contact:
            selected_contact = grpname

        total_message, total_media_shared, total_links_shared, total_words = await self.analyser.fetch_total_media(df)
        
        plots = {
            "day_timeline": await self.analyser.day_time_line(df),
            "most_busy_day": await self.analyser.most_bussy_day(df),
            "most_busy_month": await self.analyser.most_bussy_month(df),
            "heatmap": await self.analyser.dayVShour(df),
            "most_busy_users": await self.analyser.Most_bussy_Users(df),
            "wordcloud": await self.analyser.Word_Cloud(df),
            "common_words": await self.analyser.Most_common_Word(df),
            "emoji_pie": await self.analyser.emojiCountPie(await self.analyser.emojiCount(df))
        }

        busy_users_table = await self.analyser.NameVSPercentage(df)
        emoji_table = await self.analyser.emojiCount(df)

        os.makedirs(self.data_analyser_config.data_analyser_dir, exist_ok=True)
        
        for name, fig in plots.items():
            plot_path = os.path.join(self.data_analyser_config.data_analyser_dir, f"{name}.png")
            fig.savefig(plot_path)
            plt.close(fig)

        await write_file(os.path.join(self.data_analyser_config.data_analyser_dir, "busy_users.csv"), busy_users_table)
        await write_file(os.path.join(self.data_analyser_config.data_analyser_dir, "emoji_counts.csv"), emoji_table)

        stats = {
            "total_messages": total_message,
            "total_words": int(total_words),
            "media_shared": int(total_media_shared),
            "links_shared": int(total_links_shared)
        }
        await write_yml(os.path.join(self.data_analyser_config.data_analyser_dir, "stats.yaml"), stats)

        analysis_report = {
            "contacts": contacts,
            "selected_contact": selected_contact,
            "stats": stats,
            "plots": plots,
            "busy_users_table": busy_users_table,
            "emoji_table": emoji_table
        }
        
        logger_obj.info("Data Analyser Pipeline completed successfully")
        return DataAnalyserArtifact(analysis_report=analysis_report)
