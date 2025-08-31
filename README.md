# An Enterprise LLM System for HealthCare Applications

We present and enterprise LLM system that could be deployed by healthcare providing entities for knowledge discovery, project assistance, and topic aggregation. The LLM system present an interface that is similar to the ChatGPT interface through which healthcare professionals can interact with their own data warehoused in a vector database (Pinecone). 

The system imploy the use of Langchain, OpenAIEmbeddings and Pinecove vector database to convert healthcare enterprise data into vector embeddings using OpenAIEmbeddings. The vector embeddings are stored in Pinecone Cloud vector database. Streamlit is used to provide a friendly user interface through which healthcare practitioners can interact with their vectorised enterprised datasets. 

### System Architecture 
The high-level system architecture and the system front end are displayed in the figures below:

---

<img width="521" height="301" alt="Image" src="https://github.com/user-attachments/assets/db8beb9c-c908-4247-8a01-d715d6b72159" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/2d785f47-8aa3-4471-86bc-444d663c7372" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/e84b4423-1165-4209-9ffc-b88f867f2ab8" />

---


### Enterprise Data
The data warehoused in the Pinecone Vector database is an enterprise dataset since all forms of data (structure or unstructured) and most data formats (png, jpeg, txt, pdf, wav, xml, html etc) can be chuncked and uploaded into the 
into the Pinecone vector database. 

### Streamlit Front End
The Streamlit front end can be used for:

* Querying/interacting with the system (user interface)
* Uploading new data into the Pinecone vector database
* Displaying the result of the users' queries.

### Montoring System Performance with Prometheus and Grafana
Prometheus timeseries database is used to scrape important system metrics such as 

* Total queries in a time range
* Total uploaded files in a time range
* Total file upload failures in a time range 
* File upload latency
* Query latency

Grafana is used to display the results of Prometheus metrics.

## Instructor

* Luke Thompson, Ph.D.
* Assistant Research Professor, NOAA Northern Gulf Institute, Atlantic Oceanographic and Meteorological Laboratory, Miami, Florida
* Email: [luke.thompson@noaa.gov](mailto:lukethompson@noaa.gov)
* Pages: [GitHub](https://github.com/cuttlefishh), [Google Scholar](https://scholar.google.com/citations?user=kggNWsMAAAAJ), [NOAA profile](https://swfsc.noaa.gov/staff.aspx?id=22360)

## Online Content

* GitHub repository: <https://github.com/cuttlefishh/python-for-data-analysis>
* YouTube channel: <https://www.youtube.com/channel/UCVZrIrWtcvTzYlrNx7RcDyg>

## Textbooks

* [_Learn Python 3 the Hard Way_](https://learnpythonthehardway.org/python3/) by Zed Shaw (Addison-Wesley) -- Step-by-step introduction to Python with no prior knowledge assumed; includes appendix Command Line Crash Course.
* [_Learning Python_](http://proquest.safaribooksonline.com/book/programming/python/9781449355722) 3rd Edition by Mark Lutz (O'Reilly) --  Optional; more traditional introduction to Python as a computer language.


---
### Hallucination Mitigation Strategies
#### Mitigation strategies:
##### Data Quality
    (1)Use data of high quality to train LLMs. (2)Ensure the training data is accurate, comprehensive, and free from biases. (3) Ensure that the data captures the overall range of your desired objectives
##### Grounding Techniques
    Use cleansed and curated data and expose the LLM to the data's repository. This is the approach that we used in this discourse.
##### Retrieval-Augmented Generation (RAG)
    Integrate real-time knowledge retrieval methods into the response generation process. The real time data may be harvested through a website, an API endpoint, or an OLAP database that stores real time company's data.
##### Advanced prompting techniques
     Use techniques like chain-of-thought prompting to guide the LLM's reasoning. 
##### Prompt engineering
     Carefully crafting prompts to elicit more accurate and reliable responses

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/8805177c-fbea-4481-8414-82395a17f256" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/0ab1a93e-74e9-4080-8271-d57707f6d87d" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/5b588f6d-2336-4c3d-821e-66b961fae90c" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/5f226b98-4a50-4d81-8bc9-e97b2da1ad7b" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/fe1f48e8-e941-4474-9a51-074bfa84e246" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/1708aec5-67d1-4e95-9e80-c9c5e702dbf8" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/12ef7201-1a5d-4d46-8884-7219d018f5f8" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/e3393cf5-f00f-4567-ad51-e3f7171c2a82" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/baf76756-7e74-4aae-ab44-de3172d495cb" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/268cb56b-4c03-443a-b59b-d08b15e14da3" />

---

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/26b112f6-40c2-4ac2-b6d8-8e8a581181ad" />

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d57c65d5-7631-46e8-b05f-79e69024a770" />

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/bb2af9cd-e830-428d-ab65-972b5ce15573" />

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d21e0858-cb82-4deb-b1f5-ecd10de57b46" />

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/d3cac355-9e48-4e95-af6a-4724ac28f74f" />








--
Thank you for reading through

---

### **AUTHOR'S BACKGROUND**

```

Author's Name:  Emmanuel Oyekanlu
Skillset:   I have experience spanning several years in developing scalable enterprise data pipelines,
soluttion architecture, architecting enterprise systems data and AI applications, software and AI solution
design and deployments, data engineering, high performance computing (GPU, CUDA), machine learning, NLP and LLM
applications as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)
