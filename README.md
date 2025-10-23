Earthquake Ground Motion Directionality Prediction



This project builds a \*\*data processing and machine learning pipeline\*\* to predict \*\*Δθ (directionality angle difference)\*\* between seismic recording sites using waveform data, site characteristics, and earthquake metadata.



The workflow extracts per-site features, generates pairwise site data, and trains predictive models to analyze ground motion directionality across earthquake events.

DATASETS- This project uses datasets from NGA WEST.



---



\## ⚙️ Pipeline Overview



\### \*\*Step 1: Create Site-Level Dataset\*\*

\*\*Script:\*\* `create\_site\_dataset.py`  

\*\*Input:\*\*  

\- `NGAW2\_flatfile.csv` – NGA-West2 metadata  

\- `waveforms/` – directory of waveform files (`.AT2`, `.VT2`, `.DT2`)  



\*\*Output:\*\*  

\- `site\_data.csv` – contains computed per-site seismic features such as PGA, PGV, PGD, Vs30, and Δθ  



\*\*Key operations:\*\*

\- Reads waveform data per RSN (record sequence number)  

\- Extracts PGA, PGV, and PGD from multiple channels  

\- Computes mean values and directionality angle Δθ  

\- Filters and saves valid results  



Run:

python create\_site\_dataset.py

python generate\_pairs.py

python train\_models\_visual.py



Developed by Tavish Shah (2025) – Earthquake Ground Motion Directionality Prediction Pipeline

