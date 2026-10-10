# Data protection and AI governance

This document describes what the DataMind code does today with the data it is
given, and gives a first assessment of how the tool could be classified under the
EU AI Act. It is a description of the current behaviour, written from the source
code. It is not a legal opinion and it does not claim that the project complies
with the GDPR, the EU AI Act or any other law.

Items marked **to be reviewed** are uncertain or depend on facts outside this
repository (contracts, deployment setup, legal interpretation) and need to be
checked by someone qualified before the tool is used with real employee data.

## 1. What data the pipeline reads

The input is one CSV file of employee records: an upload to the API
(`POST /analyze`, `POST /analyze/stream`), an upload in the Streamlit app
(`app.py`), or `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` for `main.py`.

The whole file is loaded into memory with `pandas.read_csv`
(`agents/ingestion_agent.py`), so every column in the file is read, including any
that the analysis does not need. The analysis itself uses six columns:

| Column | Used for |
|---|---|
| `Attrition` | Attrition rate overall and per group; splitting the charts into left / stayed |
| `Department` | Attrition rate and average income per department |
| `MonthlyIncome` | Average income per department; income histogram |
| `OverTime` | Attrition rate with and without overtime |
| `JobSatisfaction` | Attrition rate per satisfaction level |
| `Age` | Average age of leavers and stayers; age histogram |

The other columns are not analysed, but their **names** (not their values) are
listed in the ingestion summary (`numerical_columns`, `categorical_columns`), which
is printed to the console and returned in the API response.

The API rejects a file that lacks any of the six columns. The Streamlit app and
`main.py` do not validate the file.

Records of real employees are personal data. `Age` and `MonthlyIncome` are
sensitive in an employment context even though they are not "special categories"
under GDPR Article 9. The pipeline does not remove or mask direct identifiers
(for example an employee number or name column) from an uploaded file; it simply
does not use them. Whether an upload contains special-category data depends on the
file and is **to be reviewed** per dataset.

## 2. What is sent to the LLM provider

The only external call that carries data is the chat completion in
`agents/insight_agent.py`, sent to OpenRouter (`https://openrouter.ai/api/v1`) with
the model set to `openrouter/auto`.

**It sends aggregated statistics, not raw rows.** The prompt built by
`build_prompt` contains exactly:

- the overall attrition rate (one percentage);
- the attrition rate per department (the department names from the CSV and one
  percentage each);
- the attrition rate per `OverTime` value (the values from the CSV and one
  percentage each);
- the average age of employees who left and of employees who stayed;
- the text retrieved from the `.txt` files in `rag_docs/` (the top three
  documents by similarity; currently one research file);
- fixed instructions for the report format.

Not sent: individual rows, the DataFrame, the average income per department, the
attrition rate per job-satisfaction level, the column list, the file name.

Points to be aware of:

- **Small groups.** No minimum group size is enforced. If a department or an
  overtime group has very few people, its rate can reveal information about
  identifiable individuals (a department of one person has a rate of 0% or 100%).
  Whether the statistics count as anonymous for a given dataset is
  **to be reviewed**.
- **Category values are sent as text.** The values of `Department` and `OverTime`
  are passed through as they appear in the CSV. If those columns contain
  unexpected free text, that text is sent.
- **The receiving model is not fixed.** `openrouter/auto` lets OpenRouter choose
  the underlying model, so the code does not determine which model provider
  processes the prompt, or in which country. OpenRouter's and the underlying
  providers' retention, logging and training-use terms, and the international
  transfer basis, are **to be reviewed**.
- **Whatever is in `rag_docs/` is sent.** Do not put confidential or personal
  material in that folder.
- The prompt and the HTML report describe the data as the "IBM HR Attrition
  dataset" regardless of which file was uploaded.

Other network activity: the embedding model (`all-MiniLM-L6-v2`) is downloaded
from Hugging Face the first time it is needed (in Docker, while the image is
built). Embedding and the FAISS search run locally and in memory; no employee
data is sent to Hugging Face.

## 3. What is stored, for how long, and how to delete it

Nothing is deleted on a schedule. There is no retention period in the code.

| What | Where | Kept until |
|---|---|---|
| Uploaded CSV (API) | A temporary file in the system temp directory | Deleted by the API when the analysis steps finish, also on failure |
| Uploaded CSV (Streamlit app) | A temporary file in the system temp directory | **Not deleted by the app.** It stays until someone or the operating system clears the temp directory |
| Charts and HTML report (API) | `outputs/<request_id>/` (4 PNG files, 1 HTML file) | **Indefinitely.** No code removes them |
| Charts (Streamlit app, `main.py`) | `outputs/`, fixed file names | Overwritten by the next run, otherwise indefinitely |
| HTML reports (Streamlit app, `main.py`) | `reports/`, one timestamped file per run | Indefinitely |
| Console output and logs | Wherever the process output goes | Depends on the deployment; **to be reviewed** |

What the stored files contain:

- The HTML report holds the overall attrition rate, the two average ages, the
  attrition rate per department and the LLM-generated text.
- Charts 1 and 2 are bar charts of the aggregated rates. Charts 3 and 4 are
  histograms of `MonthlyIncome` and `Age`, split into left / stayed. They are
  binned counts, not rows, but with a small dataset a bin can correspond to one
  person.
- The console output includes the statistics, the first 300 characters of the
  LLM text and of the retrieved context. On a failed request the API logs the
  uploaded file's name and the traceback, and returns the error message of the
  failed step to the caller; whether such messages can contain data values is
  **to be reviewed**.

The API does not serve the files in `outputs/`; it only returns their paths. The
folder name is a random ID and is not linked to who made the request.

### Deleting outputs

There is no delete endpoint. Files are removed by hand on the machine that runs
the service, from the directory the server was started in.

One request:

```
rm -rf outputs/<request_id>                          # Linux / macOS
Remove-Item -Recurse -Force outputs\<request_id>     # Windows PowerShell
```

Everything:

```
rm -rf outputs reports                               # Linux / macOS
Remove-Item -Recurse -Force outputs, reports         # Windows PowerShell
```

In Docker, outputs live in `/app/outputs` inside the container and disappear when
the container is removed (`docker run --rm` removes it on stop). The `docker run`
command in the README mounts the host's `outputs/` folder there, so with that
command the files persist on the host and are deleted with the commands above.

Deleting local files does not delete anything that OpenRouter or the underlying
model provider may have retained (see section 2).

`outputs/`, `reports/`, `data/` and `.env` are excluded from git and from the
Docker build context.

## 4. Access control

**The API has no authentication yet.** There is also no authorisation, no rate
limiting and no limit on upload size. Anyone who can reach the port can upload a
file, trigger a paid LLM call with the server's `OPENROUTER_API_KEY`, and receive
the statistics computed from that file. The Streamlit app has no login either.

`uvicorn` as started in the README serves plain HTTP. TLS, network restrictions
and access control have to be provided by the deployment and are not part of this
repository.

Until authentication is added, the service should only be run locally or on a
trusted network, and should not be exposed to the internet with real employee
data.

## 5. EU AI Act: preliminary classification

This is a first assessment to guide a proper review. The whole section is
**to be reviewed** by legal counsel. References are to Regulation (EU) 2024/1689.

### What the system does today

- It computes descriptive statistics about a workforce as a whole.
- It asks a general-purpose LLM, through a third-party API, to write findings,
  root causes and recommendations for HR management from those statistics.
- It does **not** score, rank, predict or classify individual employees, and it
  takes no decision or action itself. Its output is a report for a human reader.

### Why the high-risk category is relevant

Annex III, point 4 lists "employment, workers management and access to
self-employment" as a high-risk area. Point 4(b) covers AI systems intended to be
used to make decisions affecting the terms of work-related relationships,
promotion or termination, to allocate tasks based on individual behaviour or
personal traits, or to monitor and evaluate the performance and behaviour of
persons in such relationships. Systems that fall under Annex III are high-risk
under Article 6(2).

### Assessment

- **As built, classification is uncertain.** The tool works at the level of
  aggregate statistics and does not evaluate individuals, which argues against it
  falling under point 4(b). But it is built for HR management, and its
  recommendations are meant to influence workforce decisions (overtime policy,
  pay, retention measures in a specific department). Whether that intended
  purpose brings it within point 4(b) is **to be reviewed**.
- **It would very likely be high-risk if** it were extended to predict which
  individual employees are likely to leave, to flag or rank individuals, or if
  its output were used to inform decisions about specific people (pay, promotion,
  contract renewal, termination).
- **The Article 6(3) exception is narrow.** A system in an Annex III area may be
  treated as not high-risk if it poses no significant risk, for example because
  it only performs a preparatory task. That exception never applies to a system
  that profiles natural persons, and relying on it requires a documented
  assessment (Article 6(4)). Whether it could apply here is **to be reviewed**.
- **Roles.** Whoever places the tool on the market or puts it into service under
  their own name would be the provider; an organisation that uses it on its own
  workforce would be the deployer. The same organisation can be both. The LLM is a
  general-purpose AI model supplied by a third party. The allocation of
  obligations is **to be reviewed**.
- **Dates.** The dates on which the high-risk obligations start to apply have been
  the subject of proposed amendments. The currently applicable dates are
  **to be reviewed**.

Independently of the AI Act, processing employee data falls under the GDPR. The
legal basis, the need for a data protection impact assessment (Article 35), the
information given to employees, any works-council or employee-representative
involvement required by national law, and the processor agreement and transfer
basis for the LLM provider are all **to be reviewed**.

### Human oversight that would be needed

None of the following exists in the code today, except that the HTML report
labels the LLM section "AI-Generated Insights".

- A named, trained person reviews every report before it is used, with the
  authority to reject or disregard it (Article 14).
- The report is never the sole basis for a decision about an individual employee.
- Reviewers can see the input statistics next to the LLM text, so that claims in
  the text can be checked against the numbers. The LLM can state causes and
  figures that the data does not support; the "root causes" section in particular
  is generated explanation, not a result of causal analysis.
- A minimum group size, so that small departments are not singled out.
- A check of the statistics and recommendations for effects on protected groups.
  `Age` is used directly, and age is a protected characteristic.
- A record of each run: who ran it, on which data, which model answered, and who
  reviewed the result (Article 12). `openrouter/auto` currently makes the model
  unknown and results not reproducible.
- A way to stop using the tool and to report and investigate problems.

### Transparency that would be needed

- Employees and their representatives are told that the tool is used, on which
  data and for what purpose, before it is put into use (Article 26(7) for
  high-risk systems; GDPR Articles 13 and 14 in any case).
- Users of the tool get clear instructions: what it does, what data it needs,
  what goes to the LLM provider, and its limits (Article 13).
- Every output is clearly marked as AI-generated and as not verified. The HTML
  report labels the section; the API response and the SSE stream carry the text
  without such a marker. Any obligations under Article 50 are **to be reviewed**.
- The people who operate and use the tool have sufficient AI literacy (Article 4).
- This document is kept up to date when the code changes.

## 6. Summary of gaps

- No authentication, authorisation, rate limiting or upload size limit.
- No retention period and no automatic deletion of `outputs/` or `reports/`.
- The Streamlit app leaves the uploaded CSV in the temp directory.
- No minimum group size before statistics are sent to the LLM or reported.
- The LLM model and provider are not pinned; provider terms are not reviewed.
- No human-review step, no audit record of runs.
- No legal review of the AI Act classification or of GDPR obligations.
