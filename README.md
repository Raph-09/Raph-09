# 👋 HI, I'M RAPHAEL

A versatile Data Scientist with three years’ experience analyzing data for insights, building predictive models, and deploying these models into applications to solve business and research problems. My work spans data analysis, data science, machine learning, modern AI, and MLOps, allowing me to bridge the gap between raw data and intelligent, production-ready systems.


---

# 🛠️ SKILLS IN CATEGORIES 
## DATA ANALYSIS SKILLS
-MYSQL    
-POWERBI  
-TABLEAU <br/>
-PANDA  <br/>
-MATPLOTLIB <br/> 
-SEABORN  <br/>
-PLOTLY
## DATA SCIENCE SKILLS 
-SCIKIT-LEARN <br/>
-JUPYTER NOTEBOOK <br/>
-GOOGLE COLAB <br/>
-XGBOOST <br/>
-KERAS/TENSORFLOW <br/>
-MACHINE LEARNING <br/>
-DEEP LEARNING <br/>
-NATURAL LANGUAGE PROCESSING (NLP) <br/>

## MACHINE LEARNING ENGINEERING/MLOPS SKILLS 
-FLASK <br/>
-STREAMLIT(for quick prototyping) <br/>
-MODULAR/OBJECT ORIENTED PROGRAMMING <br/>
-MLFLOW (for experiement tracking and registration)  <br/>
-DATA VERSION CONTROL (DVC) for versioning data and model  <br/>
-DAGSHUB (Platform similar to github for hosting data, code and model versions) <br/>
-EvidentlyAI (for model monitoring) <br/>
-DOCKER <br/>
-AZURE CONTAINER REGISTRY (ACR) <br/>
-AZURE CONTAINER APP (ACA) <br/>

## GENERATIVEAI AND RAG
-LLM API: AZUREOPENAI, GROK,HUGGINGFACE, GEMINI API  <br/>
-EMBEDDINGS: HUGGINGFACE AND GEMINI EMBEDDINGS  <br/>
-VECTOR DATABASE: ASTRAL DATABASEA ND FAISS VECTOR STORE  <br/>
-ONLINE SEARCH TOOL: TAVILY SEARCH TOOL <br/>
-CREWAI (for agent development) <br/>


----I AM STILL STUDYING THIS






---

# 💼 EXPERIENCE

## Data Scientist

## BIZSTEK GLOBAL · Full-time

### Oct 2022 - Sep 2025 · 3 yrs

### Lagos State, Nigeria · Remote

### Key Responsibilities & Contributions:

1. Data Collection & Preparation: Designed and implemented data ingestion and scraping pipelines to gather high-quality datasets from diverse sources.
2. Data Cleaning & Feature Engineering:** Performed preprocessing, handling missing data, encoding categorical variables, and engineering domain-specific features to enhance model accuracy.
3. Exploratory Data Analysis (EDA):  Conducted in-depth EDA and visualization to identify hidden patterns, correlations, and actionable insights using Python (Pandas, Matplotlib, Seaborn).
4. Model Development & Evaluation:   Experimented with various machine learning algorithms (Logistic Regression, Random Forest, XGBoost, etc.) to select optimal models based on accuracy, precision, recall, and F1 score.
5. Model Deployment:  Deployed best-performing models into production environments using Flask, Docker, and Azure, ensuring scalability and seamless integration with business systems.
6. Performance Monitoring:  Applied  EvidentlyAI for monitoring data drift and model performance, maintaining reliability over time.
7 Cross-functional Collaboration: Worked closely with developers and business stakeholders to translate analytical findings into practical solutions that align with strategic objectives.


## Data Science Intern

## iNeuron.ai · Full-time

### May 2021 - Jan 2022 · 9 mos

### India · Remote

### Key Responsibilities & Contributions:

i. Recommender System Development: Designed and developed a hyper-personalized recommendation engine for an online store to improve customer engagement and sales conversion rates.

ii. Data Exploration & Wrangling: Performed extensive Exploratory Data Analysis (EDA) and data preprocessing, ensuring data consistency and quality across multiple datasets.

iii. Feature Engineering: Engineered new customer-related features and selected the most impactful variables for model training, significantly boosting performance.

iv. Algorithm Experimentation: Evaluated different recommender system algorithms, including collaborative and content-based filtering, using libraries such as surprise and scikit-learn.

v.  Simulation & Model Validation: Implemented simulations to test recommendation strategies under varying user behaviors and dataset dynamics.

vi. Research & Model Optimization: Studied and implemented machine learning best practices to fine-tune models, reduce overfitting, and improve recommendation accuracy.


---

# 🚀 Projects


### 1. **AMAZON PRODUCT RECOMMENDER CHATBOT** <br/>
The Amazon Chatbot is an AI-powered e-commerce assistant designed to answer product-related queries using a Retrieval-Augmented Generation (RAG) approach. It leverages Amazon product reviews and titles to provide contextually relevant, concise, and helpful responses to user queries.

The system integrates LangChain, HuggingFace embeddings, AstraDB vector store, and Groq LLM, with a Flask-based web interface. It is containerized with Docker and supports continuous integration/deployment via GitHub Actions and DockerHub.

A retrieval-augmented generation system integrated with Tavily web search and Groq LLM, containerized with Docker.

* 🔗 Repo: [[Link to GitHub Repository](https://github.com/Raph-09/AMAZON-PRODUCT-RECOMMENDER-CHATBOT)]


### 2. **RAG with Web Search Functionality**

The Web-Search AI Assistant is an intelligent, real-time question-answering system designed to provide accurate, up-to-date, and contextually relevant responses. Unlike traditional static language models, this application integrates live web search through Tavily with high-performance AI reasoning using the Groq LLM.

The system is built with:

LangChain for orchestrating LLM and search tools
Groq AI models for ultra-fast and efficient inference
Tavily Search API for retrieving real-time web information
Flask for providing a simple and responsive web interface
To support reliable and scalable deployment, the application is containerized using Docker and features a Continuous Integration and Deployment (CI/CD) pipeline that automatically pushes the container image to DockerHub upon updates.

This makes the system suitable for production-ready environments, research use, and applications requiring fresh, internet-retrieved knowledge.



* 🔗 Repo: [[Link to GitHub Repository](https://github.com/Raph-09/RAG-with-web_search-functionality)]

### 3. **Predictive Maintenance with MLOPS**

The purpose of this application is to build a **predictive maintenance system** capable of forecasting machine failures before they occur. It implements a full **MLOps pipeline** to ensure that data processing, model development, deployment, monitoring, and automation are performed in a scalable, reproducible, and production-ready manner. The system supports continuous integration, versioning, containerization, cloud deployment, and real-time monitoring of model performance.

* 🔗 Repo: [[Link to GitHub Repository](https://github.com/Raph-09/Predictive-Maintenance-with-MLOps)]

### 4. **Adversarial Machine Learning for Intrusion Detection**

This experiment was performed to determine the effectiveness of the proposed defence technique against evasion attacks. The IDS model was built using the CICIDS 2017 dataset found on open-source platforms like Kaggle.com. This dataset is chosen because it is used extensively by several researchers for conducting similar research (Usama et al. 2017; Haroon et al. 2022). The dataset was collected using the Python programming language and downloaded into a notebook environment for processing and analysis. After the dataset was prepared, it was used to train a decision tree algorithm to develop an IDS for detecting network intrusion. Three Different attacks were used to test the vulnerability and robustness of the trained decision tree model, of which the performance deteriorated greatly. The proposed defence of adversarial training with an ensemble classifier was put to the test by training the Adaboost, an ensembled technique, on the generated perturbated examples, and the model recorded improved performance on both seen and unseen perturbations.

* 🔗 Repo: [[Link to Colab Notebook]([https://github.com/Raph-09/dversarial_Machine_learning_Framework_for_Enhancing_the_Robustness_of_IDS_against_Evasion_Attacks./blob/main/Adversarial_Machine_learning_Framework_for_Enhancing_the_Robustness_of_IDS_against_Evasion_Attacks.ipynb](https://colab.research.google.com/drive/1XKRZLEHR6Mrs0EzDERGVBJdrmnhI3nQg?usp=sharing))]

### 5. **RepurposeAI  App using Gemini api**

Repurpose AI is an application that transforms long-form blog content into platform-ready social media posts using artificial intelligence. It helps users quickly generate tailored content for platforms like LinkedIn, Instagram, and Facebook with minimal manual editing.

The app uses Streamlit for an easy-to-use web interface, while Google Generative AI handles text generation and embeddings to ensure the posts stay relevant to the original content. To support fast and efficient content retrieval, Repurpose AI uses FAISS as its vector database. Overall, it simplifies content repurposing and helps creators maintain a consistent online presence.

* 🔗 Repo: [[Link to GitHub Repository](https://github.com/Raph-09/RepurposeAI-RAG-Application)]

  
### 6. **Phishing Websites Monitoring System (with mlops)**

1. Introduction
The Phishing Websites Monitoring System is an AI-driven application designed to detect and classify phishing websites by leveraging machine learning techniques. The project implements a full ML lifecycle pipeline, covering data ingestion, preprocessing, transformation, model training, evaluation, deployment, and monitoring.

This system enhances network security by identifying potential phishing threats using a robust model training and tracking infrastructure, supported by MongoDB, Flask, Docker, MLflow, and DagsHub.

2. Objectives
Automate the process of ingesting and processing phishing-related data.
Build and validate robust machine learning models for phishing website detection.
Enable real-time monitoring of model performance.
Provide a web-based interface for batch predictions.
Ensure scalability, reproducibility, and traceability using Docker, MLflow, and GitHub.

* 🔗 Repo: [[Link to GitHub Repository](https://github.com/Raph-09/Phishing-Webites-Monitoring-System-with-MLOps)]


# 📚 Research Contributions

* **Adversarial ML for Intrusion Detection:** Designed and implemented adversarial training frameworks using boosting ensembles.
* **Explainable AI for Predictive Maintenance:** Integrated SHAP and LIME to build transparent industrial AI systems.
* **LLM-Based Document Processing:** Leveraged Google Gemini for document understanding tasks.

---
