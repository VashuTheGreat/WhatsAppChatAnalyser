# WhatsApp Chat Analyser

A comprehensive tool to analyze WhatsApp chat exports and visualize messaging patterns, user activity, and linguistic trends. Now enhanced with an **AI-Powered Chat Assistant** for interactive data exploration.

## 🚀 New Features

- **🤖 AI Chat Assistant:** Interactive floating chat widget powered by **LangGraph** and **Groq (Llama 3.3)**. Ask questions about your chat data in natural language!
- **📊 Markdown Support:** Beautifully formatted AI responses with tables, lists, and bold text using `marked.js`.
- **Top Statistics:** Total messages, total words, media shared, and links shared.
- **Activity Timelines:** Monthly and daily activity trends.
- **Activity Maps:** Identification of the busiest days and months.
- **Weekly Heatmap:** Detailed hourly activity distribution across the week.
- **User Engagement:** Analysis of the most active participants with percentage contributions.
- **Linguistic Analysis:** Word clouds and frequency analysis of the most common words.
- **Emoji Analysis:** Most used emojis with counts and a distribution pie chart.
- **Searchable Dropdown:** Easily filter and select specific contacts for individual analysis using a searchable interface.

## 📁 Project Structure

- `api/`: Web application interface built with FastAPI and Jinja2 templates.
- `src/WhatsApp_Analyser/`: Core logic organized into components and pipelines.
    - `pipelines/ai_pipeline.py`: Handles interactive LLM-based analysis.
    - `graphs/`: LangGraph state machine definition.
    - `tools/`: Python code execution environment for AI data analysis.
- `artifacts/`: Generated analysis reports and intermediate data.
- `config/`: Configuration files for data validation and processing.

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/WhatsAppChatAnalyser.git
   cd WhatsAppChatAnalyser
   ```

2. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Install dependencies (using uv):**
   ```bash
   uv sync
   ```

4. **Run the application:**
   ```bash
   uv run main.py
   ```

## 💡 Usage

1. Export your WhatsApp chat as a `.txt` file (without media).
2. Upload the file through the web interface.
3. Once the analysis dashboard appears, click the **🤖 icon** in the bottom-right corner.
4. Chat with the AI about your data (e.g., *"Who sent the most messages on weekends?"*).

## 🧰 Technologies Used

- **AI/LLM:** LangGraph, Groq (Llama 3.3), LangChain
- **Backend:** FastAPI, Python
- **Frontend:** Bootstrap 5, Tom Select, Marked.js (Markdown Rendering)
- **Data Processing:** Pandas, NLTK, Scikit-learn
- **Visualization:** Matplotlib, WordCloud, Seaborn
