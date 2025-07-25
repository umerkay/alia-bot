
## Live
Visit https://huggingface.co/spaces/umerkk164/alia-health

## How the AI Agent Works

This medical health chatbot uses an intelligent multi-agent architecture to provide comprehensive answers about patients by accessing and analyzing different data sources. Here's how it works:

### 🏥 Patient Data Sources

The system has access to four key types of patient data:

- **📋 Assessments**: Measurement-based care assessments taken by patients that provide objective and self-reported insights into their mental health state
- **🏥 EHR (Electronic Health Records)**: Complete medical history including diagnoses, medications, and family medical history
- **📝 Intake Forms**: Patient responses to clinician-designed intake assessments (customizable per clinician)
- **💬 Session Transcripts**: Detailed transcripts from therapy sessions showing therapeutic progress and interactions

### 🤖 Multi-Agent Architecture

The system uses a **hierarchical agent structure** with specialized agents for different data types:

#### Main Clinical Agent (Orchestrator)
- Acts as the central coordinator that receives clinician queries
- Intelligently routes questions to appropriate specialized agents
- Has access to all tools and can call multiple agents in a single response
- Provides comprehensive, structured final answers

#### Specialized Sub-Agents

1. **📊 Assessment Agent**
   - Specializes in analyzing measurement-based care assessments
   - Interprets scores, trends, and clinical significance
   - Tracks patient progress over time
   - Provides context about assessment tools and clinical thresholds

2. **💭 Transcript Agent**
   - Focuses on therapy session analysis
   - Retrieves and analyzes session content
   - Provides insights into therapeutic progress
   - Cites specific sessions and interactions

3. **🏥 EHR Agent (Graph RAG)**
   - Uses advanced graph database technology for structured medical data
   - Handles three specific data types:
     - `diagnoses`: Patient conditions and diseases
     - `medications`: Current and past medications with dosages
     - `family_history`: Relevant family medical history

### 🔍 Smart Retrieval Technology

The system uses two advanced retrieval methods:

1. **Vector Search (Semantic RAG)**: For unstructured text data like transcripts, assessments, and intake forms
   - Uses Google's Generative AI embeddings for semantic understanding
   - Finds contextually relevant information even when exact keywords don't match

2. **Graph RAG**: For structured EHR data
   - Uses Neo4j graph database for complex medical relationships
   - Enables precise queries about diagnoses, medications, and family history
   - Maintains relationships between different medical entities

### 🎯 Query Processing Flow

1. **Query Reception**: Clinician asks a question about a patient
2. **Intelligent Routing**: Main agent analyzes the query and determines which data sources are needed
3. **Parallel Processing**: Multiple specialized agents can be called simultaneously
4. **Data Retrieval**: Each agent uses its specialized retrieval tool to find relevant information
5. **Integration**: Main agent synthesizes all retrieved information
6. **Clinical Response**: Provides a comprehensive, structured answer with proper medical context

### 💡 Key Features

- **Context-Aware**: Maintains conversation history for follow-up questions
- **Patient-Specific**: All queries are filtered by patient ID for data security
- **Multi-Modal**: Can handle both structured (EHR) and unstructured (transcripts, assessments) data
- **Real-Time**: Provides streaming responses with tool execution visibility
- **Safety-First**: Only uses factual data, never hallucinates information

### 🔧 Technical Implementation

- **Framework**: LangChain + LangGraph for agent orchestration
- **LLM**: Google Gemini 2.0 Flash for natural language processing
- **Vector Database**: ChromaDB for semantic search
- **Graph Database**: Neo4j for EHR relationship mapping
- **Embeddings**: Google Generative AI embeddings for semantic understanding

This architecture ensures that clinicians receive accurate, comprehensive, and contextually relevant information about their patients while maintaining strict data security and medical accuracy standards.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+:**  Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)
*   **pip:** Python's package installer (usually included with Python).

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/umerkayvyro/alia-bot/
    cd alia-bot
    ```

2.  **Create a conda environment (recommended):**

    ```bash
    conda create --name langchain python=3.9 # or your desired Python version
    conda activate langchain
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**

    *   Create a `.env` file in the root directory of the project:
        
        ```bash
        cp .env.example .env
        ```

    *   Edit the `.env` file and add your configuration:

        ```
        GOOGLE_API_KEY=<Your_Google_API_Key>
        GOOGLE_MODEL=gemini-2.0-flash
        TEMPERATURE=0.5
        
        # Path Configuration (customize as needed)
        CHROMA_DB_PATH=shared_chroma_db
        SHARED_DOCS_PATH=shared_docs
        BM25_INDEX_PATH=bm25_index.pkl
        ```

        *   Replace `<Your_Google_API_Key>` with your actual Google API key.  You'll need to obtain this from the Google Cloud Console.
        *   `GOOGLE_MODEL`: Specifies the Gemini model to use (default: `gemini-2.0-flash`).
        *   See `ENVIRONMENT_VARIABLES.md` for a complete list of available configuration options.
        *   `TEMPERATURE`: Controls the randomness of the model's output (default: `0.5`).  Lower values make the output more predictable, higher values make it more creative.

    **Important:**  Never commit your `.env` file to version control.  Add it to your `.gitignore` file.

Absolutely — here’s a clean, GitHub-friendly Markdown version of your project structure for your `README.md`:

## Project Structure
```
alia-chatbot/
├── app/                   # Main application directory
│   ├── **pycache**/       # Python cache files (ignored)
│   ├── models/            # Data models (e.g., for chat messages)
│   │   └── chat\_models.py
│   ├── public/            # Static files (e.g., HTML, CSS, JavaScript)
│   │   └── index.html
│   ├── routes/            # API route definitions
│   │   ├── **pycache**/
│   │   ├── chat.py        # Chat-related API endpoints
│   │   └── home.py        # Home/index route
│   ├── services/          # Business logic and external service integrations
│   │   ├── **pycache**/
│   │   └── chat\_service.py # Logic for interacting with the Gemini model
│   ├── utils/             # Utility functions and helpers
│   ├── config.py          # Configuration settings
│   ├── dependencies.py    # Dependency injection setup
│   └── main.py            # FastAPI application entry point
├── .env                   # Environment variables (API keys, settings)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```
## Running the Application

1.  **Navigate to the project root directory in your terminal.**

2.  **Run the application using Uvicorn:**

    ```bash
    uvicorn app.main:app --reload
    ```

    *   `app.main`: Specifies the module containing the FastAPI application.
    *   `app`:  Specifies the variable name of the FastAPI instance within `main.py`.  If your app instance has a different name, adjust accordingly.
    *   `--reload`: Enables automatic reloading on code changes (for development).

3.  **Access the application in your browser:**

    *   Open your web browser and go to `http://127.0.0.1:8000` (or the address shown in the Uvicorn output).

## API Endpoints TODO: MORE DETAIL

*   **`/` (GET):**  Serves the `index.html` file (likely the chatbot interface). Defined in `routes/home.py`.
*   **`/chat` (POST):**  Handles chat requests.  Takes user input and sends it to the Gemini model via the `chat_service`. Defined in `routes/chat.py`.

## Key Components

*   **`app/main.py`:**  The main entry point for the FastAPI application.  It initializes the FastAPI app, sets up middleware, and includes the API routers.
*   **`app/routes/chat.py`:** Defines the `/chat` API endpoint.  It receives user messages, passes them to the `chat_service`, and returns the model's response.
*   **`app/services/chat_service.py`:** Contains the logic for interacting with the Gemini model.  It handles authentication, sends requests to the model, and processes the responses.
*   **`app/config.py`:**  Loads configuration settings from environment variables using the `Settings` class.
*   **`.env`:** Stores sensitive information like API keys and configuration settings.

## Configuration

The application is configured using environment variables.  The following variables are used:

*   `GOOGLE_API_KEY`:  **Required.**  Your Google API key.
*   `GOOGLE_MODEL`:  The Gemini model to use (default: `gemini-2.0-flash`).
*   `TEMPERATURE`: The temperature setting for the model (default: `0.5`).

## Deployment

The application can be deployed to various platforms, including:

*   **Google Cloud Platform (GCP):**  Use Cloud Run or App Engine.
*   **Heroku:**  A popular platform-as-a-service.
*   **AWS:**  Use Elastic Beanstalk or ECS.
*   **Docker:**  Containerize the application for easy deployment.

**Example Deployment to Google Cloud Run:**

1.  **Create a Dockerfile:**

    ```dockerfile
    FROM python:3.9-slim-buster

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
    ```

2.  **Build the Docker image:**

    ```bash
    docker build -t alia-chatbot .
    ```

3.  **Push the image to Google Container Registry (GCR):**

    ```bash
    docker tag alia-chatbot gcr.io/<your_gcp_project_id>/alia-chatbot
    docker push gcr.io/<your_gcp_project_id>/alia-chatbot
    ```

4.  **Deploy to Cloud Run:**

    ```bash
    gcloud run deploy --image gcr.io/<your_gcp_project_id>/alia-chatbot --platform managed --region <your_gcp_region>
    ```

    *   Replace `<your_gcp_project_id>` with your Google Cloud project ID.
    *   Replace `<your_gcp_region>` with your desired Google Cloud region.

5.  **Set the `GOOGLE_API_KEY` environment variable in Cloud Run.**  You can do this through the Cloud Console or using the `gcloud` command-line tool.

## Contributing

Contributions are welcome!  Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Write tests for your changes.
5.  Submit a pull request.
