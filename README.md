# An Enterprise LLM System for Healthcare Applications

We present an enterprise large language model  (LLM) system that could be deployed by healthcare practitioners for knowledge discovery, project assistance, and  healthcare topic aggregation. The LLM system features a user interface (UI) that is similar to the ChatGPT UI through which healthcare professionals can interact with, and query their own data warehoused as vector embeddings in a vector database (Pinecone). 

### The system is composed of the following core components:

<ins>**LangChain**</ins> – Manages orchestration and workflow across the system components.


<ins>**OpenAIEmbeddings**</ins>  – Converts enterprise healthcare data into high-dimensional vector representations.

<ins>**GPT-4 (LLM)**</ins>  – Provides reasoning, natural language understanding, and advanced query handling.

<ins>**Pinecone Vector Database**</ins>  – Stores and indexes embeddings in the cloud to enable efficient similarity search and retrieval.

<ins>**Streamlit User Interface**</ins>  – Delivers an interactive, user-friendly frontend through which healthcare practitioners can query and analyze their vectorized enterprise datasets.

<ins>**Prometheus**</ins>  time series database and <ins>**Grafana**</ins>  are used as overall system performance monitoring tools. 

We also discuss how an Enterprise Architect may work alongside the Solution Architect to use MLOps (machine learning operations) best practices to scale up the system in an enterprise setting using MLOps and CI/CD orchestration tools such as ZenML, Kubeflow, Airflow and Kubernetes. 

### System Architecture 
The high-level system architecture and the system front end are displayed in the figures below:

---

<img width="521" height="301" alt="Image" src="https://github.com/user-attachments/assets/695c1c01-95b5-4657-9b93-7d274248e7a6" />

---
### System Streamlit Front-End

Examples of how the system provide answers through the Streamlit front end are provided in the series of figures below:

---
#### Query: Highlight the relationships between University of Maryland and MALDI
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/98a0bd01-c2c9-45f9-a19c-c77bd31e3be2" />

---
#### Response from LLM system
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/da314dd8-f483-4d5b-a84a-3a3c918b08fa" />

---
#### Further response from LLM system
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/fef22d10-1952-4d81-b60e-b70dceb0ecd2" />

---
#### Query: How can trials meet the NIH definition of clinical trials? Cite references.
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d1fe77e1-5bbd-47f0-87f9-32c2435d2938" />

---
#### Response from LLM system
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/09e7d95b-90e5-4ce5-bb0d-327a7ed8d4ae" />

---

#### The Streamlit front end can be used for:

* Querying/interacting with the system (user interface)
* Uploading new data into the Pinecone vector database
* Displaying the result of the users' queries.


### Enterprise Data
The data warehoused in the Pinecone Vector database is an enterprise dataset since all forms of data (structured or unstructured) and most data formats (png, jpeg, txt, pdf, wav(sound), xml, docx, eml, wav html etc) can be chuncked and uploaded into the Pinecone vector database. 


### Montoring System Performance with Prometheus and Grafana
Prometheus timeseries database is used to scrape important system metrics such as: 

##### Query Latencies

*  upload_latency_seconds → upload time

*  query_latency_seconds → full query time

*  retrieval_latency_seconds → Pinecone retrieval only

*  llm_latency_seconds → GPT generation only

  ##### File/Data Upload Failures

*  upload_failures_total

*  retrieval_failures_total

*  llm_failures_total

*  query_failures_total

##### System Concurrency

* queries_in_progress

*  retrievals_in_progress

*  llms_in_progress

*  Document Size

*  document_chunks_uploaded (histogram of how many chunks each doc upload produced)

  ##### Users' Related Metrics

* total_queries{user="emmanuel"}

* query_latency_seconds_bucket{user="anonymous"}

* document_chunks_uploaded_sum{user="emmanuel"}

#### Grafana is used to display the results of metrics that are logged by Prometheus.

##### A figure showing query latency as logged by Prometheus is shown below:

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/021ebcd4-5531-440b-bfe2-53255cf42bba" />

---

##### A figure showing the LLM latency as captured  by Grafana from Prometheus  is shown below:

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/e18ce078-fd8a-47a3-8059-3fe6feff8670" />







# How to Deploy the Project for Healthcare and other Enterprise Applications

The project can be deployed for numerous other enterprise applications even though this iteration is for health-care use case. To deploy for other applications, the enterprise dataset in the Pinecone vector database should be augmented to include the intended use case data.  As an example, a use case of the project for manufacturing use case is available here:

If the Solution Architect does not have access keys to a particular Pinecone database, the Solution Architect can just create a new index on Pinecone, and then warehouse the vector embeddings of the new dataset on the newly created Pinecone index.

### Create Pinecone Vector Database
As mentioned earlier, to be able to deploy the solution for other applications in enterprise settings, the Solution Architect must first create a Pinecone database that warehouses the vector embeddings of the dataset of the desired application. Infographs detailing how the Solution Engineer may set up a free instance of the Pinecone vector database are provided below: 

---

#### Create a new index on Pinecone
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/9795a732-4745-457a-84c3-215e06007a1d" />

---

#### Click on Customs settings and set the vector dimension to 1536. (1536 is the OpenAI default vector length for most of OpenAI data embedding models)
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d75d5dd2-546c-450e-818f-275e534413a1" />

---

#### Now, click to create the index
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/7116f7de-58c6-4eb5-aa99-97e5b0587eae" />

---

#### The new index will be available under the set of indexes available on your Pinecone homepage
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/c38337cf-0662-4675-bc4f-a9b95a17a1c8" />


### Setting up the project for On-Prem or Cloud Deployments
This project has been developed using VSCode on a Windows 10 enterprise system equipped with GeForce 3070 GPU. Without loss of generality, other types of commodity computers with other variants of operating systems can be used to deploy the application. The project structure on VSCode can be set up as shown below: 

---

<img width="201" height="426" alt="Image" src="https://github.com/user-attachments/assets/c01b744a-9d89-4db4-aa8d-c881d2588524" />



To deploy the project, the Solution Engineer must clone the project from the Github repository here:  https://github.com/manuelbomi/An-Enterprise-LLM-System-for-HealthCare-Applications.git  

A variant of the deployment for manufacturing use case can also be cloned from here: 

---
#### Example of the main.py running on VSCode
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/4ede62a2-7903-43f0-b916-a4dc732de390" />

---

After cloning, the project can be run using **docker-compose up --build** . The complete system (Streamlit + Prometheus + Grafana) should then be available on Docker. The Pinecone database is deployed on Pinecone cloud at [www.pinecone.io](https://app.pinecone.io). The Pinecone API key and the Pinecone Environment keys are used to connect the LLM system with the Pinecone vector database. 


OpenAI API key is used to obtain key that facilitate the usage of OpenAPI Embeddings. Langchain is used to connect the OpenAPIEmbedding and the vectorized healthcare data to the Pinecone vector database. 


---
> [!IMPORTANT]
> ### Docker must be available and running on the local system before the **docker-compose up --build** command
> ### Virtual environment should be created for the project, and it must be activated prior to running the project.
> ### The .env file must be on the same directory as the main.py. The .env file must contain the Pinecone API key, the OpenAI API key, the Pinecone Environment, and Pinecone Index Name
> ### The .gitignore must be in the root directory and it must contain the virtual environmnet name and the .env

---

### Scaling Up for Enterprise Applications
#### Using Callable Functions
The Enterprise Architect may desire to deploy the system for other use cases in the enterprise. It will be observed that many important segments of the code in the main.py are developed as callable functions. Using callable functions will enable the Solution or Enterprise Architect to easily deploy the system for other applications. Also, using callable functions is fundamental to good MLOps and CI/CD pratices since adjuastement can easily be made. Also, productionising will be easy and robust after any adjustment.  

#### Using ZenML or Airflow for MLOps
Orcestration tools such as ZenML or Airflow can be used together with CI/CD tools such as Github Actions. Using ZenML or Airflow will aid MLOps strategies across the enterprise. 

#### Kubeflow for deploying on Kubernetes 
As usesrs of the system increases, there may be a need to scale up the number of Kubernetes pods and Docker nodes that are used in production. Using tools such as Kubeflow as orchestrating tool will enable such scaling up to be easily acheived. This will lead to a very reactive and robust system that seamlessly serves its enterprise objectives.   








--
Thank you for reading through

---
### **AUTHOR'S BACKGROUND**
### Author's Name:  Emmanuel Oyekanlu
```
Skillset:   I have experience spanning several years in data science, developing scalable enterprise data pipelines,
enterprise solution architecture, architecting enterprise systems data and AI applications,
software and AI solution design and deployments, data engineering, high performance computing (GPU, CUDA), machine learning,
NLP and LLM applications as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)




















































