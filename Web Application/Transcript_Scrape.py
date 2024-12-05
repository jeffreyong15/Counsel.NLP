import pandas as pd
from datetime import datetime

def scrape_transcript(FILE):
    print(FILE)
    df = pd.read_html(FILE, header=None)[0]
    df.drop(columns=['Description','Grade', 'Units', 'Grd Points', 'Repeat Code','Reqmnt Desig', 'Status'], inplace=True)
    df[['Semester', 'Year']] = df['Term'].str.split(' ', expand=True)
    df['Year'] = df['Year'].astype(int)
    df.drop('Term', axis=1, inplace=True)

    df.to_csv('Transcript.csv', index=False)
    