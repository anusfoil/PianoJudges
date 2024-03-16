import subprocess
import os
import json
import csv
import glob
import pandas as pd
import numpy as np
from youtube_search import YoutubeSearch
import hook

def should_download(title, duration, filter_keywords, max_duration=10000000000):
    """Check if the video should be downloaded based on the title and duration.
    
    Args:
        title (str): The title of the video.
        duration (int): The duration of the video in seconds.
        filter_keywords (list): A list of keywords to filter out.
        max_duration (int): The maximum duration allowed for a video in seconds.
        
    Returns:
        bool: True if the video should be downloaded, False otherwise.
    """
    # Check title against filter keywords
    for keyword in filter_keywords:
        if keyword.lower() in title.lower():
            return False
    
    # Check duration
    if duration > max_duration:
        return False
    
    return True

def download_channel_videos(url, filter_keywords):
    # Command to fetch video information
    if "channel" in url or "user" in url:
        command = ['yt-dlp', '--dump-json', '--verbose', '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '--flat-playlist', url]
    else:
        command = ['yt-dlp', '--dump-json', '--verbose', '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', url]

    # Get video information without downloading
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    videos_info = result.stdout.splitlines()


    for video_info in videos_info:
        video_json = json.loads(video_info)
        title = video_json.get('title', '')
        duration = video_json.get('duration', 0)  # Get the duration in seconds
        video_url = 'https://www.youtube.com/watch?v=' + video_json['id']

        # Check if the video should be downloaded based on title and duration
        if should_download(title, duration, filter_keywords):
            # Download command
            download_command = [
                'yt-dlp',
                '--format', 'bestaudio/best',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '16',
                '--write-description',
                '--write-info-json',
                '--write-comments',
                '--ignore-errors',
                '--output', os.path.join(DATA_DIR, '%(id)s.%(ext)s'),
                video_url
            ]
            subprocess.run(download_command)


def write_metadata_to_csv():
    metadata_files = glob.glob(os.path.join(DATA_DIR, '*.info.json'))

    with open(os.path.join(DATA_DIR, 'metadata.csv'), 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'mark_sub_level', 'title', 'upload_date', 'duration', 'view_count', 'like_count', 'dislike_count', 'comment_count', 'description', 'webpage_url', 'channel_url', 'comments'])

        for metadata_file in metadata_files:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

                comments = ''
                if metadata.get('comments'):
                    try:
                        comments = [c['text'] for c in metadata.get('comments')]
                    except:
                        pass
                writer.writerow([
                    metadata.get('id'),
                    '',  # my custom field for my notes
                    metadata.get('title'), 
                    metadata.get('upload_date'), 
                    metadata.get('duration'),
                    metadata.get('view_count'), 
                    metadata.get('like_count'), 
                    metadata.get('dislike_count'),
                    metadata.get('comment_count'), 
                    metadata.get('description'),
                    metadata.get('webpage_url'),
                    metadata.get('channel_url'),
                    comments
                ])
    
    # make a copy of the metadata file
    category = DATA_DIR.split('/')[-1]
    os.system(f"cp {DATA_DIR}/metadata.csv {category}_metadata.csv")
    print(f"{category}_metadata.csv")



def cleanup_artifacts():
    # File extensions to delete
    extensions_to_delete = ['*.webp', '*.jpg', '*.json', '*.description']

    # Delete files with the specified extensions
    for extension in extensions_to_delete:
        for file in glob.glob(os.path.join(DATA_DIR, extension)):
            os.remove(file)




def delete_unreferenced_wav_files(metadata_path, audio_dir, category):
    # Load the updated metadata
    updated_metadata = pd.read_csv(metadata_path)
    referenced_files = list(updated_metadata['id'])
    referenced_files = set([f'/import/c4dm-datasets/PianoJudge/{category}/' + path + '.wav' for path in referenced_files])

    # Get all .wav files in the directory
    wav_files = set(glob.glob(os.path.join(audio_dir, '*.wav')))

    # Determine which .wav files are not referenced in the metadata
    unreferenced_files = wav_files - referenced_files

    # Delete the unreferenced .wav files
    for file_path in unreferenced_files:
        os.remove(file_path)
        print(f"Deleted: {file_path}")


def recover_download(csv_file_path, output_directory):
    """
    Download audio files from YouTube based on IDs in a CSV file.

    Parameters:
    - csv_file_path: Path to the CSV file containing YouTube video IDs.
    - output_directory: Directory where the .wav files will be saved.
    """

    # Read the YouTube IDs from the CSV file
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        youtube_ids = [row['id'] for row in reader if int(row['duration']) < 400] # only download videos less than 5min (a bit tolerance)

    for youtube_id in youtube_ids:
        # Construct the YouTube URL
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        # Define the yt-dlp command to download the audio as .wav
        command = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',  # Convert audio to wav format
            '--audio-quality', '0',  # Best audio quality
            '-o', f'{output_directory}/%(id)s.%(ext)s',  # Output template
            url  # YouTube URL
        ]

        # Execute the download command
        try:
            subprocess.run(command, check=True)
            print(f"Downloaded and converted {youtube_id} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {youtube_id}: {e}")


def delete_large_embeddings(directory):
    """
    Delete .npy audio embeddings with the first dimension size greater than 400.

    Parameters:
    - directory: The directory to scan for .npy files.
    """
    
    # Loop through all files in the specified directory
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            file_path = os.path.join(directory, file)
            # embedding = np.load(file_path)
            # Check the size of the first dimension
            if '.npy' in file_path:
            # if embedding.shape[0] > 400:
                # Delete the file if the first dimension is greater than 400
                os.remove(file_path)
                print(f"Deleted {file_path} due to size constraint.")

def remove_duplicates(metadata_path):

    # Load the metadata CSV file
    df = pd.read_csv(metadata_path)

    # Drop duplicates based on the 'id' column
    df_no_duplicates = df.drop_duplicates(subset=['id'], keep='first')

    # Save the cleaned DataFrame to a new CSV file
    df_no_duplicates.to_csv(metadata_path, index=False)

    print(f"Cleaned metadata saved to {metadata_path}")




# This function constructs search queries based on your specifications
def construct_queries(metadata):
    queries = []
    for item_id, item_details in metadata.items():
        composer = item_details["composer"]
        work_name = item_details["work_name"]
        num_paths = len(item_details["path"])
        if num_paths > 1:
            for movement in range(1, num_paths + 1):
                query = f'{composer} {work_name} mvt. {movement}'
                queries.append((composer, work_name, f'mvt. {movement}', query))
        else:
            query = f'{composer} {work_name}'
            queries.append((composer, work_name, '', query))
    return queries

def fetch_youtube_links(query):
    print(query)
    # search_response = youtube.search().list(
    #     q=query,
    #     part='id,snippet',
    #     maxResults=3,
    #     type='video',
    #     videoDuration='short'
    # ).execute()

    results = YoutubeSearch(query, max_results=20).to_dict()

    links = []
    for search_result in results:
        if len(links) > 3:
            break
        if int(search_result['duration'].split(':')[0]) < 5:
            suffix = search_result['url_suffix'].split('&')[0]
            video_link = f'https://www.youtube.com{suffix}'
            links.append(video_link)

    return links


def cipi_json_to_csv():

    # Assuming `metadata` is your JSON data
    metadata = json.load(open('/homes/hz009/Research/PianoJudge/data_collection/index.json'))

    queries = construct_queries(metadata)

    # Writing results to a CSV file
    with open('CIPI_youtube_links.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Composer', 'Work Title', 'Movement', 'YouTube Link 1', 'YouTube Link 2', 'YouTube Link 3'])

        for composer, work_name, movement, query in queries:
            youtube_links = fetch_youtube_links(query)
            writer.writerow([composer, work_name, movement] + youtube_links)

def match_henle():
    metadata = json.load(open('/homes/hz009/Research/PianoJudge/data_collection/index.json'))
    cipi = pd.read_csv("/import/c4dm-datasets/PianoJudge/difficulty_cipi/CIPI_youtube_links.csv")
    split = json.load(open('/homes/hz009/Research/PianoJudge/data_collection/splits.json'))
    split0 = split['0']

    # cipi['henle'] = 0
    cipi['id'] = ''
    cipi['split'] = ''
    for mt, data in metadata.items():
        cipi.loc[(cipi['Composer'] == data['composer']) & (cipi['Work Title'] == data['work_name']), 'id'] = mt
        if mt in split0['train']:
            sp = 'train'
        elif mt in split0['val']:
            sp = 'val'
        elif mt in split0['test']:
            sp = 'test'        
        cipi.loc[(cipi['Composer'] == data['composer']) & (cipi['Work Title'] == data['work_name']), 'split'] = sp
        # cipi.loc[(cipi['Composer'] == data['composer']) & (cipi['Work Title'] == data['work_name']), 'henle'] = data['henle']

    cipi.to_csv("/import/c4dm-datasets/PianoJudge/difficulty_cipi/CIPI_youtube_links.csv", index=False)
    return 


if __name__ == "__main__":

    category = 'difficulty_cipi'

    # DATA_DIR = "/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/"
    # url_file = "/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/urls_all.list"
    # DATA_DIR = f"/import/c4dm-datasets/PianoJudge/{category}/"
    # url_file = f"/homes/hz009/Research/PianoJudge/data_collection/{category}_channels.txt"

    # DATA_DIR = "/import/c4dm-datasets/PianoJudge/techniques/"
    # url_file = f"/homes/hz009/Research/PianoJudge/data_collection/technique_groups.txt"

    DATA_DIR = "/import/c4dm-datasets/PianoJudge/difficulty_cipi/"
    # url_file = f"/homes/hz009/Research/PianoJudge/data_collection/mikrokosmos.txt"

    # delete_unreferenced_wav_files(DATA_DIR + "metadata.csv", DATA_DIR, category)
    # recover_download(DATA_DIR + "metadata.csv", DATA_DIR)
    # delete_large_embeddings(DATA_DIR)
    # remove_duplicates(DATA_DIR + "metadata.csv")
    # hook()
    # channel_url = 'https://www.youtube.com/@nixxpiano'

    # Define keywords to filter out, and download
    filter_keywords = ['concerto', 'duets']
    # with open(url_file) as f:
    #     urls = f.readlines()
    urls = pd.read_csv('/import/c4dm-datasets/PianoJudge/difficulty_cipi/CIPI_youtube_links.csv')['YouTube Link 1']
    for url in urls:
        if url[0] != '#': # not commented out
            id = url.split('v=')[-1].strip()
            if 'shorts' in id:
                id = id.split('shorts/')[-1].strip()
            if os.path.exists(DATA_DIR + id + '.wav'):
                print(f"Skipping {id}")
                continue
            download_channel_videos(url, filter_keywords)

    # write_metadata_to_csv()
    cleanup_artifacts()

    
