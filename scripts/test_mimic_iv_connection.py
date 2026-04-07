import getpass
import requests

def test_mimic_iv_connection():
    print("============================================================")
    print("  MIMIC-IV v3.1 CONNECTION TESTER (Direct PhysioNet)")
    print("============================================================")
    print("Testing connection to: https://physionet.org/files/mimiciv/3.1/\n")
    
    username = input("PhysioNet Username (or Email): ").strip()
    password = getpass.getpass("PhysioNet Password: ")
    
    # PhysioNet data file server URL for MIMIC-IV v3.1
    url = "https://physionet.org/files/mimiciv/3.1/"
    
    print("\nVerifying credentials...")
    
    # We use a session with the exact same headers that succeeded for the images
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'Wget/1.21.2'})
    
    try:
        # A simple lightweight HEAD/GET request to the directory root
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            print("\n✅ CONNECTED SUCCESSFULLY!")
            print("Your credentials are fully authorized for MIMIC-IV v3.1.")
            print("You are ready to download the Hosp, ICU, or Derived files whenever you need to.")
        elif response.status_code == 401:
            print("\n❌ UNAUTHORIZED (HTTP 401): Incorrect username or password.")
        elif response.status_code == 403:
            print("\n❌ ACCESS DENIED (HTTP 403): Your account does not have access.")
            print("Please ensure you have signed the Data Use Agreement specifically for MIMIC-IV v3.1")
            print("at: https://physionet.org/content/mimiciv/3.1/")
        elif response.status_code == 404:
            print("\n❌ NOT FOUND (HTTP 404): The PhysioNet path was not found.")
        else:
            print(f"\n⚠️ FAILED: Unexpected HTTP {response.status_code} response.")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ NETWORK ERROR: Failed to reach PhysioNet servers. Details:\n{e}")

if __name__ == "__main__":
    test_mimic_iv_connection()
