try:
    from Websites.youtube import YouTubeDetector
except ImportError:
    from BotDetector.Websites.youtube import YouTubeDetector

import time

import pandas as pd


def test():

    df = pd.DataFrame({"text": ["A", "B", "C", "D"]})
    index = 0
    for i, row in df.iterrows():
        for j, nrow in df.iterrows():
            if i == j:
                pass
            else:
                print(f"I_{index}: {row['text']}    {nrow['text']}")
            index += 1


if __name__ == "__main__":

    url = "https://youtube.com/shorts/qYcVT3BAQJA?si=McC5WjFirpPaW_JB"
    start = time.time()
    bot_detector = YouTubeDetector(url)

    # occurrences = bot_detector.keyword_search("2017")
    # print(f"Number: {occurrences}")
    matches = bot_detector.get_text_similarity()
    matches.to_csv("./test.csv")
    # print(f"Occurrences: {occurrences}")

    end = time.time()
    elapse = end - start
    print(f"Elapse: {elapse}")
