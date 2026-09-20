"""Data engineering: turn raw scraper metadata into downloaded, sanity-checked
images (notebook 01's deterministic steps — id slugging, image download,
corruption check). EDA-only code (size/resolution/color-distribution stats,
brand-count plots) stays inline in the notebook, since it's not part of the
pipeline contract.
"""
import time
import random
import re
import unicodedata

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def slugify(text):
    '''
    Converts a string into a clean, consistent format suitable for use in an ID:
    lowercase, accents removed, special characters stripped, spaces replaced with underscores.

    For example:
      "100% Pure"        -> "100_pure"
      "e.l.f. cosmetics" -> "elf_cosmetics"
      "L'Oréal"          -> "loreal"
      "Clé de Peau"      -> "cle_de_peau"
    '''
    if pd.isna(text): return ''
    text = str(text).lower().strip()
    # decompose accented characters (e.g. é -> e + combining accent), then drop the accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # remove anything that isn't a letter, number, or space (%, ', &, ., /, etc.)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # collapse whitespace and replace with underscores
    text = re.sub(r'\s+', '_', text.strip())
    return text


def download(df, dest_folder: str):
  '''
  This function takes the images from the URL and saves them.
  The filename is derived from the unique product ID rather than the URL,
  to avoid collisions when different brands use generic image names like
  'swatch.jpg'. The file extension is still taken from the URL.
  If an URL was broken, the error is saved in img_name for later inspection.
  '''

  df['img_name'] = None
  saved = 0
  failed = 0

  for i in tqdm(range(len(df)), desc='Downloading images'):
    url = df.loc[i, 'img_url']
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}

    try:
      r = requests.get(url, headers=headers, timeout=10)

      if r.ok:
        # derive extension from URL, stripping query params first; default to jpg if unclear
        last_segment = url.split('/')[-1].split('?')[0]
        file_type = last_segment.split('.')[-1][0:3] if '.' in last_segment else 'jpg'
        # use the unique product ID as filename to avoid collisions across brands
        file_name = df.loc[i, 'id']
        file_path = dest_folder + '/' + file_name + '.' + file_type

        df.loc[i, 'img_name'] = file_name + '.' + file_type
        with open(file_path, 'wb') as f:
          f.write(r.content)
        saved += 1

      else:  # HTTP status code 4XX/5XX
        df.loc[i, 'img_name'] = r.status_code
        failed += 1

    except requests.exceptions.RequestException as e:
      df.loc[i, 'img_name'] = 'error'
      failed += 1

    # random pause between requests to avoid getting blocked
    time.sleep(random.uniform(0.5, 2.0))

  print(f"Done. Saved: {saved} | Failed: {failed}")
  return df


def check_img(filename):
    '''
    Checks whether a file is a valid, uncorrupted image.

    Opens the file twice: once to verify the header and integrity,
    and once to attempt a pixel-level operation (flip), which catches
    files that pass header checks but are truncated or malformed.

    Returns None if the file is valid, or a string describing the error
    if the file is missing, not an image, or corrupted (e.g. an HTML
    response saved as an image file).
    '''
    try:
        im = Image.open(filename)
        im.verify()
        im.close()
        im = Image.open(filename)
        im.transpose(Image.FLIP_LEFT_RIGHT)
        im.close()
        return None  # no error
    except Exception as e:
        return str(e)  # return the reason
