from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from tqdm import tqdm
import re
import concurrent.futures
BASE_URL = "https://www.shl.com"
CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"
# Setup Chrome driver (for individual driver instances)
def init_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
# Scrape individual assessment detail page (run concurrently)
def scrape_detail(url):
    driver = init_driver()
    try:
        driver.get(url)
        # Reduce sleep time for faster scraping; adjust if needed
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        #Duration Extraction
        duration = "N/A"
        # Use a CSS selector to directly find a <p> tag with "approximate completion time" text.
        duration_tag = soup.select_one("p:contains('approximate completion time')")
        if not duration_tag:
            # Fallback: search all <p> tags
            duration_tags = soup.find_all("p")
            for tag in duration_tags:
                if "approximate completion time" in tag.get_text().lower():
                    duration_tag = tag
                    break
        if duration_tag:
            match = re.search(r"(\d+)", duration_tag.get_text())
            if match:
                duration = int(match.group(1))
        #Test Type Extraction
        type_mapping = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations"
        }
        type_codes = []
        type_block = soup.find(text=re.compile("test type", re.I))
        if type_block:
            container = type_block.find_parent()
            if container:
                found_codes = re.findall(r"\b([A-Z])\b", container.get_text(" ").strip())
                for code in found_codes:
                    if code in type_mapping:
                        type_codes.append(type_mapping[code])
        test_type = ", ".join(type_codes) if type_codes else "N/A"
        #Remote / Adaptive Flags
        text_content = soup.get_text(separator="\n").lower()
        remote = "Yes" if "remote testing" in text_content else "No"
        adaptive = "Yes" if "adaptive" in text_content or "irt" in text_content else "No"
        return {
            "Duration": duration,
            "Test Type": test_type,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {
            "Duration": "N/A",
            "Test Type": "N/A",
            "Remote Testing Support": "No",
            "Adaptive/IRT Support": "No"
        }
    finally:
        driver.quit()
#Scrape a single listing page to get assessment URLs and titles
def scrape_page(url):
    driver = init_driver()
    driver.get(url)
    # Lower the sleep time to speed up; adjust if your connection is slow.
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = soup.find_all("a", href=True)
    assessments = []
    for link in links:
        href = link["href"]
        if "/product-catalog/view/" in href:
            title = link.get_text(strip=True)
            full_link = BASE_URL + href if href.startswith("/") else href
            assessments.append({
                "Assessment Name": title,
                "URL": full_link
            })
    driver.quit()
    return assessments
#Scrape all pages (first scrape listings, then concurrently fetch details)
def scrape_all_pages():
    all_assessments = []
    #Scrape type=1 pages
    for start in tqdm(range(0, 384, 12), desc="Scraping Type 1 pages"):
        url = f"{CATALOG_URL}?start={start}&type=1"
        all_assessments.extend(scrape_page(url))
    #Scrape type=2 pages
    for start in tqdm(range(0, 144, 12), desc="Scraping Type 2 pages"):
        url = f"{CATALOG_URL}?start={start}&type=2"
        all_assessments.extend(scrape_page(url))
    #Remove duplicate assessments based on URL
    unique_assessments = {}
    for item in all_assessments:
        unique_assessments[item["URL"]] = item
    all_assessments = list(unique_assessments.values())
    #Now concurrently scrape details for each assessment
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {executor.submit(scrape_detail, item["URL"]): item for item in all_assessments}
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item),
                           desc="Scraping Details"):
            item = future_to_item[future]
            details = future.result()
            item.update(details)
    return all_assessments
#Save the scraped metadata to CSV and JSON files
def save_metadata(data):
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    df.to_csv("shl_metadata_index.csv", index=False)
    with open("shl_metadata_index.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Metadata saved to CSV and JSON.")
if __name__ == "__main__":
    print("Starting SHL scraper...")
    all_assessments = scrape_all_pages()
    print(f"Total assessments found: {len(all_assessments)}")
    save_metadata(all_assessments)
