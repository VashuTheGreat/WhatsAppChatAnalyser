# WhatsApp Chat Analyser

A comprehensive tool to analyze WhatsApp chat exports and visualize messaging patterns, user activity, and linguistic trends.

## Features

- **Top Statistics:** Total messages, total words, media shared, and links shared.
- **Activity Timelines:** Monthly and daily activity trends.
- **Activity Maps:** Identification of the busiest days and months.
- **Weekly Heatmap:** Detailed hourly activity distribution across the week.
- **User Engagement:** Analysis of the most active participants with percentage contributions.
- **Linguistic Analysis:** Word clouds and frequency analysis of the most common words.
- **Emoji Analysis:** Most used emojis with counts and a distribution pie chart.
- **Searchable Dropdown:** Easily filter and select specific contacts for individual analysis using a searchable interface.

## Project Structure

- `api/`: Web application interface built with FastAPI and Jinja2 templates.
- `src/WhatsApp_Analyser/`: Core logic organized into components and pipelines (Ingestion, Validation, Transformation, Analysis).
- `artifacts/`: Generated analysis reports and intermediate data.
- `config/`: Configuration files for data validation and processing.
- `logger/` & `exception/`: Standardized logging and custom exception handling.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/WhatsAppChatAnalyser.git
   cd WhatsAppChatAnalyser
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   # OR
   uvicorn api.app:app --reload
   ```

## Usage

1. Export your WhatsApp chat as a `.txt` file (without media for faster analysis).
2. Upload the file through the web interface.
3. Use the searchable dropdown to filter statistics by specific contacts or view the overall group analysis.

## Technologies Used

- **Backend:** FastAPI, Python
- **Frontend:** HTML5, Bootstrap 5, Tom Select (Searchable Dropdown)
- **Data Processing:** Pandas, NLTK
- **Visualization:** Matplotlib, WordCloud
