import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("YOUTUBE_API")


import time
import re

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

from pytubefix import YouTube
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# NLP
from sentence_transformers import SentenceTransformer

# Scikit learn
from sklearn.metrics.pairwise import cosine_similarity


class YouTubeDetector:
    def __init__(self, url: str) -> None:

        self.yt = YouTube(url)
        self.title = self.yt.title
        self.video_id = self.yt.video_id

        self.comment_path = f"./Data/youtube/{self.title}.csv"

    def get_comments(self, sort_by_date: bool = False):

        try:
            comments = pd.read_csv(self.comment_path).drop("Unnamed: 0", axis=1)
        except FileNotFoundError:
            comments = self.query_comments(save=True)

        if sort_by_date:
            comments["published"] = pd.to_datetime(comments["published"])

            # Sort the DataFrame by the 'published' column (ascending or descending)
            comments = comments.sort_values(
                by="published", ascending=True
            )  # Use False for descending order

            # Reset index after sorting if needed
            comments.reset_index(drop=True, inplace=True)

        return comments

    def query_comments(self, save: bool = False, part="snippet", max_results=100):
        youtube = build("youtube", "v3", developerKey=api_key)

        try:
            # Retrieve comment thread using the youtube.commentThreads().list() method

            comments = []
            next_page_token = None

            while True:
                # Retrieve a page of comment threads
                response = (
                    youtube.commentThreads()
                    .list(
                        part=part,
                        videoId=self.video_id,
                        textFormat="plainText",
                        maxResults=max_results,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                # Process the response and extract comment details
                for item in response["items"]:
                    comment_text = item["snippet"]["topLevelComment"]["snippet"][
                        "textDisplay"
                    ]
                    comment_text = comment_text.replace("\n", "").replace("\n\n", "")
                    likes = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                    user_name = item["snippet"]["topLevelComment"]["snippet"][
                        "authorDisplayName"
                    ]
                    published_timestamp = item["snippet"]["topLevelComment"]["snippet"][
                        "publishedAt"
                    ]
                    user_id = item["snippet"]["topLevelComment"]["snippet"][
                        "authorChannelId"
                    ]["value"]
                    comments.append(
                        {
                            "comment": comment_text,
                            "num_of_likes": likes,
                            "user_name": user_name,
                            "user_id": user_id,
                            "published": published_timestamp,
                        }
                    )

                # Check if there is a next page token, if not break the loop
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            if comments:
                df = pd.DataFrame(comments)
                # Sort dataframe by number of likes in descending order
                df = df.sort_values(by=["num_of_likes"], ascending=False)
            else:
                df = pd.DataFrame()  # Empty Dataframe

            df.reset_index(inplace=True)
            df.drop("index", axis=1, inplace=True)

            if save:
                df.to_csv(
                    f"./Data/youtube/{self.title}.csv",
                )

            return df

        except HttpError as error:
            print(f"An HTTP error {error.http_status} occurred:\n {error.content}")
            return None

    def get_text_similarity(self):
        comments_df = self.get_comments()
        comments_df = comments_df[
            ~comments_df["comment"].apply(self.contains_only_emojis)
        ]
        comments_df.reset_index(inplace=True, drop=True)

        comments = comments_df["comment"].to_list()
        # comments = [
        #     comment for comment in comments if not self.contains_only_emojis(comment)
        # ]
        # Load pre-trained model
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        # Encode all comments into embeddings at once
        embeddings = model.encode(comments)
        # Compute the cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        # Optionally, extract pairwise similarities in a more readable format
        n_comments = len(comments)
        matches = {
            "source_user": [],
            "source_text": [],
            "target_user": [],
            "target_text": [],
            "similarity": [],
        }
        for i in range(n_comments):
            for j in range(
                i + 1, n_comments
            ):  # Ensure we don't compare the same pair twice
                sim = similarity_matrix[i][j]
                if sim > 0.5:
                    matches["source_text"].append(comments_df["comment"].iloc[i])
                    matches["source_user"].append(comments_df["user_name"].iloc[i])
                    matches["target_text"].append(comments_df["comment"].iloc[j])
                    matches["target_user"].append(comments_df["user_name"].iloc[j])
                    matches["similarity"].append(sim)
        matches_df = pd.DataFrame(matches)
        matches_df.sort_values(by="similarity", inplace=True, ascending=False)
        matches_df.reset_index(inplace=True, drop=True)
        return matches_df

    # Function to detect if a string contains only emojis

    def contains_only_emojis(self, text):
        emoji_pattern = re.compile(
            r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+$",
            flags=re.UNICODE,
        )
        return bool(emoji_pattern.match(text))

    def keyword_search(self, keyword: str):
        comments = self.get_comments()
        # Perform keyword search (case insensitive)
        keyword_count = comments[
            comments["comment"].str.contains(keyword, case=False, na=False)
        ]

        # Get the count of rows where the keyword appears
        count = len(keyword_count)

        comment_count = len(comments)

        return (count / comment_count) * 100

    def find_bots(self):

        comments = self.get_comments()

        similar_text = []
        start = time.time()
        for i, source_row in comments.iterrows():
            source_text = source_row["comment"]
            for j in range(i + 1, len(comments)):
                target_row = comments.iloc[j]
                target_text = target_row["comment"]
                similarity = self.get_text_similarity(source_text, target_text)

                if similarity > 50:
                    similar_text.append(
                        {
                            "source_text": source_text,
                            "source_user_name": source_row["user_name"],
                            "source_user_id": source_row["user_id"],
                            "target_text": target_text,
                            "target_user_name": target_row["user_name"],
                            "target_user_id": target_row["user_id"],
                            "similarity": similarity,
                        }
                    )
        end = time.time()
        elapse = end - start

        df = pd.DataFrame(similar_text)
        df.to_csv("./test2.csv")
        print(f"DF: {df}")
        print(f"Elapse: {elapse}")
