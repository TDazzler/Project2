import os
import requests
import praw
import mimetypes
from urllib.parse import urlparse

# Utility to determine if URL is likely an image
def is_image(url):
    mimetype, _ = mimetypes.guess_type(url)
    return mimetype and mimetype.startswith('image')


# Your credentials
REDDIT_CLIENT_ID = 'y4EaUwKFr61-KdwiM94ogg'
REDDIT_CLIENT_SECRET = 'bmcigvTmC6wl1WTS7NUjxzZ2_IjWXg'
REDDIT_USER_AGENT = 'ImageDownloader by /u/TerryBhoy'

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

output_dir = 'irishwolfhound_images'
os.makedirs(output_dir, exist_ok=True)

subreddit = reddit.subreddit('irishwolfhound')
posts = list(subreddit.top(limit=1000))  # Adjust limit as needed
print(f"Found {len(posts)} posts.")

count = 0
for post in posts:
    try:
        submission = reddit.submission(id=post.id)  # fetch full submission
        print(f"Checking: {submission.url}")

        if getattr(submission, 'is_gallery', False) and hasattr(submission, 'media_metadata'):
            for item in submission.media_metadata.values():
                url = item['s']['u'].replace('&amp;', '&')
                try:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        filename = os.path.basename(urlparse(url).path)
                        image_path = os.path.join(output_dir, filename)
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded from gallery: {filename}")
                        count += 1
                except Exception as e:
                    print(f"Failed to download gallery image {url}: {e}")
            continue

        # For normal single-image posts
        url = submission.url
        if is_image(url):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    filename = os.path.basename(urlparse(url).path)
                    image_path = os.path.join(output_dir, filename)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                    count += 1
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    except Exception as e:
        print(f"Failed to process submission {post.id}: {e}")

print(f"Finished. Downloaded {count} images.")
