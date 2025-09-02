# An Enterprise LLM System for Healthcare Applications

We present an enterprise large language model  (LLM) system that could be deployed by healthcare practitioners for knowledge discovery, project assistance, and  healthcare topic aggregation. The LLM system features a user interface (UI) that is similar to the ChatGPT UI through which healthcare professionals can interact with, and query their own data warehoused as vector embeddings in a vector database (Pinecone). 

The system imploy the use of: Langchain, OpenAIEmbeddings, an LLM model (**gpt-4**),  and Pinecone vector database to convert enterprise healthcare data into vector embeddings using OpenAIEmbeddings. The vector embeddings are stored in a Pinecone Cloud vector database. Streamlit is used to provide a friendly UI through which healthcare practitioners can interact with their vectorised enterprise datasets. 

Prometheus time series database and Grafana are used as overall system performance monitoring tools. 

We also discuss how an Enterprise Architect may work alongside the Solution Engineer to use MLOps (machine learning operations) best practices to scale up the system in an enterprise setting using MLOps and CI/CD orchestration tools such as ZenML, Kubeflow, Airflow and Kubernetes. 

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
The data warehoused in the Pinecone Vector database is an enterprise dataset since all forms of data (structured or unstructured) and most data formats (png, jpeg, txt, pdf, wav(sound), xml, docx, eml, wav html etc) can be chuncked and uploaded into the into the Pinecone vector database. 


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

#### Grafana is used to display the results of Prometheus metrics.

##### A figure showing some metrics  scraped by Prometheus is shown below:

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/021ebcd4-5531-440b-bfe2-53255cf42bba" />

---

##### A figure showing some metrics  captured  by Grafana from Prometheus  is shown below:

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/e18ce078-fd8a-47a3-8059-3fe6feff8670" />

---

### How to Deploy the Project in Enterprise Settings

The complete system (Streamlit + Prometheus + Grafana) is deployed on a Docker network. The Pinecone database is deployed on Pinecone cloud at [www.pinecone.io](https://app.pinecone.io). The Pinecone API key and the Pinecone Environment keys are used to connect the local system with the Pinecone vector database. 




---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/c38337cf-0662-4675-bc4f-a9b95a17a1c8" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/9795a732-4745-457a-84c3-215e06007a1d" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d75d5dd2-546c-450e-818f-275e534413a1" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/7116f7de-58c6-4eb5-aa99-97e5b0587eae" />

---


OpenAI API key is used to obtain key that facilitate the usage of OpenAPI Embeddings. Landchain is used to connect the OpenAPIEmbedding and the vectorized healthcare data to the Pinecone vc=ector database. 



### How to Deploy for Enterprise Applications

The project can be deployed for numerous enterprise applications even though this iteration is for health-care uses. To deploy for other applications, only the enterprise dataset in the Pinecone vector database should be changed. For example, a use case of the same applications for manufacturing is available here: 

To deploy for other applications in industrial settings, the Solution Engineer must first create a Pinecone databases that warehouses the vector embeddings of the dataset of the desired application. An infograph tutorial on how to set up a free instance of the Pinecone vector database is available here: 

#### Setting up the project locally
This project has been developed using VSCode on a Windows 10 based enterprise system equipped with GeForce 3070 GPU. Without loss of generality, other types of commodity computers with other variants of operating systems can be used to deploy the application. The project structure on VSCode can be set up as shown below: 

---

<img width="201" height="426" alt="Image" src="https://github.com/user-attachments/assets/c01b744a-9d89-4db4-aa8d-c881d2588524" />

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/4ede62a2-7903-43f0-b916-a4dc732de390" />

---



### Scaling Up for Enterprise Applications
#### Callable Functions

#### ZenML (MLOps)

#### Kubeflow for deploying on Kubernetes 








--
Thank you for reading through

---
### **AUTHOR'S BACKGROUND**
### Author's Name:  Emmanuel Oyekanlu
```
Skillset:   I have experience spanning several years in developing scalable enterprise data pipelines,
data science applications, enterprise solution architecture, architecting enterprise systems data and AI applications,
software and AI solution design and deployments, data engineering, high performance computing (GPU, CUDA), machine learning,
NLP and LLM applications as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)
>>>>>>> 64a231af6d953b967a93a9c3dc4b5f052de76119






















