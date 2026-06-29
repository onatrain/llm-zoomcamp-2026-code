import requests

FAQ_BASE_URL = "https://datatalks.club/faq"
MAIN_FAQ_URL = "/json/courses.json"


class RawFaq():
    @property
    def items(self) -> list[dict[str, str]]:
        _faq_items = self._read_items()
        return _faq_items

    def _read_items(self) -> list[dict[str, str]]:
        response = requests.get(f"{FAQ_BASE_URL}{MAIN_FAQ_URL}")
        response.raise_for_status()

        main_index: list[dict[str, str]] = response.json()

        course_urls = [j["path"] for j in main_index]

        faq_items: list[dict[str, str]] = []

        for url in course_urls:
            response = requests.get(f"{FAQ_BASE_URL}{url}")
            response.raise_for_status()

            course_faq_items = response.json()

            faq_items.extend(course_faq_items)

        return faq_items
