import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the version history page
url = "https://apexlegends.wiki.gg/wiki/Version_History"

# Send a GET request
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all list items containing patch notes
patches = soup.find_all("li")

data = []

for patch in patches:
    # Extract date and determine if it's a major patch
    bold_tag = patch.find("b")
    link = patch.find("a")
    
    if bold_tag and link:
        date = link.text.strip()
        is_major = 1
        is_minor = 0
    elif link:
        date = link.text.strip()
        is_major = 0
        is_minor = 1
    else:
        date = patch.text.strip()
        is_major = 0
        is_minor = 1
    
    # Default content categorization (placeholder, can be improved)
    content, balance, other = 0, 0, 0  
    
    # Append extracted data
    data.append([date, is_minor, is_major, content, balance, other])

# Create a DataFrame
columns = ["Date", "Minor", "Major", "Content", "Balance", "Other"]
df = pd.DataFrame(data, columns=columns)

# Save to an Excel file
excel_filename = "apex_patch_notes.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Data saved to {excel_filename}")
