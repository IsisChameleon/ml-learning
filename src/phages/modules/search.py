import requests
from bs4 import BeautifulSoup
import datetime

def fetch_arxiv_papers(keyword, years=3):
    """
    Fetches a list of PDF URLs from arXiv.org related to the given keyword, published in the last specified number of years.

    :param keyword: Keyword to search for.
    :param years: Number of years to go back from current date. Default is 3 years.
    :return: List of PDF URLs.
    """

    # Current year and target year
    current_year = datetime.datetime.now().year
    target_year = current_year - years

    # URL components
    base_url = "https://arxiv.org"
    search_url = f"{base_url}/search/?query={keyword}&searchtype=all&abstracts=show&order=-announced_date_first&size=50"

    # Send request to arXiv
    response = requests.get(search_url)
    if response.status_code != 200:
        print("Failed to retrieve data from arXiv")
        return []

    # Parse the response content
    soup = BeautifulSoup(response.content, 'html.parser')
    entries = soup.find_all('li', class_='arxiv-result')
    
    # Filter and collect PDF URLs
    pdf_urls = []
    for entry in entries:
        # Get publication year
        pub_date = entry.find('p', class_='is-size-7').text
        pub_year = int(pub_date.split()[-1])

        # Check if publication year is within the target range
        if pub_year >= target_year:
            pdf_link = entry.find('a', title='Download PDF')['href']
            pdf_urls.append(f"{base_url}{pdf_link}")

    return pdf_urls


