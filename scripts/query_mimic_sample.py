import requests
import pandas as pd
import getpass

def query_mimic_without_saving():
    print("============================================================")
    print("  MIMIC-IV LIVE STREAMING QUERY (No Files Saved to Disk)")
    print("============================================================")
    
    username = input("PhysioNet Username: ").strip()
    password = getpass.getpass("PhysioNet Password: ")
    
    # Let's query the 'patients.csv.gz' file located in the 'hosp' module natively!
    url = "https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"
    
    print("\nConnecting to PhysioNet and streaming data in real-time...")
    
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'Wget/1.21.2'})
    
    # Use stream=True so we don't accidentally download massive files all at once
    response = session.get(url, stream=True)
    
    if response.status_code == 200:
        # Pandas allows us to read directly from a live HTTP stream!
        # By setting `nrows=10000`, we strictly pull the first 10,000 rows into RAM
        # and then instantly disconnect without downloading the rest of the file.
        df = pd.read_csv(response.raw, compression='gzip', nrows=10000)
        
        print(f"✅ Successfully streamed {len(df)} rows directly into memory!\n")
        
        # Now, you can query this data purely in Python (SQL-style)
        print("QUERY: Show me 5 female patients who died (dod is not null):")
        
        # Filter dataframe
        query_result = df[(df['gender'] == 'F') & (df['dod'].notna())].head(5)
        print("\n", query_result[['subject_id', 'gender', 'anchor_age', 'dod']])
        
    else:
        print(f"Failed to stream! HTTP Status: {response.status_code}")

if __name__ == "__main__":
    query_mimic_without_saving()
