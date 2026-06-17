import requests

faq_base_url = "https://datatalks.club/faq"
main_faq_url = "/json/courses.json"

response = requests.get(f"{faq_base_url}{main_faq_url}")
response.raise_for_status()

main_index = response.json()

course_urls = [j["path"] for j in main_index]

faq_items = []

for url in course_urls:
    response = requests.get(f"{faq_base_url}{url}")
    response.raise_for_status()

    course_faq_items = response.json()

    faq_items.extend(course_faq_items)
    print(len(course_faq_items))

print("There are", len(faq_items), "items in the faq.")


