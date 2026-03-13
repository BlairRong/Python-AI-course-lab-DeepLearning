# download_dataset.py - Enhanced Pixabay download with multiple keywords

import os
import requests
from pathlib import Path
import time

# --- Configuration -------------------------------------------------
# Your Pixabay API key
PIXABAY_API_KEY = "54968097-7a3bec13379d0c957df5e9345" #my_PIXABAY_API_KEY_HERE

# Define multiple search keywords for each class (to increase diversity)
# Using a list of synonyms/related phrases
CLASS_KEYWORDS = {
    'red_cup': [
        'red cup', 'red coffee mug', 'red tumbler', 'red tea cup',
        'red ceramic cup', 'red drinkware'
    ],
    'blue_bottle': [
        'blue water bottle', 'blue plastic bottle', 'blue glass bottle',
        'blue drink bottle', 'blue bottle on table'
    ],
    'phone': [
        'smartphone', 'mobile phone on table', 'iphone', 'android phone',
        'cell phone', 'phone screen'
    ]
}

# How many images to attempt to download per class (after de-duplication)
IMAGES_PER_CLASS_TARGET = 120  # I'll try to get more than needed incase the images is not relevent

# Save directory base
BASE_DIR = Path(__file__).parent / "downloaded_images"
# -------------------------------------------------------------------

def download_images_for_class(class_name, keywords, target_count):
    """Download images using multiple keywords, avoiding duplicates by image ID."""
    save_dir = BASE_DIR / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded_ids = set()  # store image IDs to avoid duplicates
    total_downloaded = 0
    page = 1
    per_page = 50

    # Loop through keywords until we reach target or run out of results
    for keyword in keywords:
        print(f"\n>>> Searching '{class_name}' with keyword: '{keyword}'")
        page = 1
        while total_downloaded < target_count:
            params = {
                'key': PIXABAY_API_KEY,
                'q': keyword,
                'image_type': 'photo',
                'per_page': per_page,
                'page': page,
                'safesearch': 'true',
                'orientation': 'horizontal'  # prefer landscape
            }

            try:
                response = requests.get("https://pixabay.com/api/", params=params)
                if response.status_code != 200:
                    print(f"  API error: {response.status_code}")
                    break

                data = response.json()
                hits = data.get('hits', [])
                if not hits:
                    print(f"  No more images for keyword '{keyword}'")
                    break

                for img in hits:
                    if total_downloaded >= target_count:
                        break

                    img_id = img['id']
                    if img_id in downloaded_ids:
                        continue  # skip duplicate

                    img_url = img['webformatURL']
                    # Download image
                    img_resp = requests.get(img_url, stream=True)
                    if img_resp.status_code == 200:
                        file_path = save_dir / f"{class_name}_{img_id}.jpg"
                        with open(file_path, 'wb') as f:
                            for chunk in img_resp.iter_content(1024):
                                f.write(chunk)
                        downloaded_ids.add(img_id)
                        total_downloaded += 1
                        print(f"  Downloaded ({total_downloaded}/{target_count}): {file_path.name}")
                    else:
                        print(f"  Failed to download: {img_url}")

                page += 1
                time.sleep(0.5)  # gentle pause to avoid hitting rate limits

            except Exception as e:
                print(f"  Error during download: {e}")
                break

        # if we reached target, stop trying more keywords
        if total_downloaded >= target_count:
            break

    print(f"\nFinished '{class_name}': downloaded {total_downloaded} unique images.\n")
    return total_downloaded

# Main execution
if __name__ == "__main__":
    print("Starting enhanced dataset download from Pixabay...")
    for class_name, keywords in CLASS_KEYWORDS.items():
        download_images_for_class(class_name, keywords, IMAGES_PER_CLASS_TARGET)
    print("\nAll downloads completed! Now you can manually clean the dataset.")
    
    
    

#run the file 
#python download_dataset.py