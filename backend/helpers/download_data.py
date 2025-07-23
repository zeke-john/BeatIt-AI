import os
import subprocess
import sys
artists = [
    # "1080PALE",
    "Dillygotitbumpin",
    "bvtman",
    "8eenmusic", 
    "AGRENIUS",
    "angeloimani",
    "baileydaniel",
    "bardobeats",
    "BeatsByCon",
    "beatsbycryptic",
    "beatsbydaku",
    "bvtman",
    "cmb_cworld",
    "datboigetro",
    "DonnieKatana",
    "eeryskies",
    "flowersInNarnia",
    "folinbeats",
    "HomagesHiddenGems",
    "isaiahtruth",
    "JACKPOT",
    "juno",
    "JustDa1",
    "justdanbeats",
    "KAREBEATS",
    "kidjoeybeats",
    "kingochops",
    "KINGXMAGS",
    "kulture.",
    "KYRIGO",
    "LegionbeatsAndHooks",
    "LethalNeedle",
    "lowtyde",
    "MANUEL",
    "mjNichols",
    "noevdvv",
    "OceanBeatsYT",
    "oddstatusbeats",
    "offszn",
    "offsznbeats",
    "OthelloBeats",
    "P.SOUL",
    "pilotbouttofly",
    "PK",
    "prodbyandyr",
    "prodbysyndrome",
    "prodobilus",
    "ProdOuiLele",
    "ProdTyBeats",
    "SamHigo",
    "Stoic",
    "svnx0",
    "SXINT",
    "teamrossdancehall",
    "terch",
    "TINYGBEATS12",
    "TUNGA",
    "UrbanNerdBeats",
    "WhoKares",
    "YMAR",
    "youngswisherbeats",
    "yvngfrvr",
    "zypizypi"
]

def create_artist_folders_and_download():
    """Create folders for each artist and run yt-dlp commands"""
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Create artists subdirectory
    artists_dir = os.path.join(data_dir, "artists")
    if not os.path.exists(artists_dir):
        os.makedirs(artists_dir)
        print(f"Created directory: {artists_dir}")
    
    # Process each artist
    for i, artist in enumerate(artists, 1):
        print(f"\n--- Processing artist {i}/{len(artists)}: {artist} ---")
        
        # Create artist folder
        artist_folder = os.path.join(artists_dir, artist)
        if not os.path.exists(artist_folder):
            os.makedirs(artist_folder)
            print(f"Created folder: {artist_folder}")
        else:
            print(f"Folder already exists: {artist_folder}")
        
        # Prepare the yt-dlp command
        output_path = os.path.join(artist_folder, "%(title)s.%(ext)s")
        youtube_url = f"https://www.youtube.com/@{artist}/videos"
        
        # yt-dlp command to download audio as MP3
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "--embed-metadata",
            "--add-metadata",
            "--format", "bestaudio",  # Changed from mp3 to bestaudio
            "--sleep-requests", "5",  # Reduced sleep between requests
            "--fragment-retries", "4",  # Add fragment retry limit
            "--retry-sleep", "10",  # Sleep between retries
            "--no-playlist-reverse",  # Don't reverse playlist order
            "--ignore-errors",  # Skip unavailable videos
            "--no-continue",  # Don't resume partially downloaded files
            "-o", output_path,
            youtube_url
        ]
        
        # Execute the command
        try:
            print(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            
            # Print output in real-time
            while process.stdout is not None:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get the return code
            return_code = process.poll()
            
            if return_code == 0:
                print(f"✓ Download successful for {artist}")
            else:
                print(f"✗ Download failed for {artist} with return code {return_code}")
                if process.stderr is not None:
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        print(f"Error: {stderr_output.strip()}")
                    
        except Exception as e:
            print(f"✗ Exception occurred while processing {artist}: {str(e)}")
        
        print(f"Completed processing for {artist}")

if __name__ == "__main__":
    print("Starting artist folder creation and download process...")
    print(f"Total artists to process: {len(artists)}")
    
    # Check if yt-dlp is available
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ yt-dlp is available: {result.stdout.strip()}")
        else:
            print("✗ yt-dlp is not available or not working properly")
            print("Please make sure you have activated the 'beatit' conda environment")
            sys.exit(1)
    except FileNotFoundError:
        print("✗ yt-dlp is not installed or not in PATH")
        print("Please run: conda activate beatit && pip install yt-dlp")
        sys.exit(1)
    
    # Check if ffmpeg is available (required for audio conversion to MP3)
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ffmpeg is available for audio conversion")
        else:
            print("✗ ffmpeg is not working properly")
            print("Please run: conda activate beatit && conda install ffmpeg -y")
            sys.exit(1)
    except FileNotFoundError:
        print("✗ ffmpeg is not installed or not in PATH")
        print("Please run: conda activate beatit && conda install ffmpeg -y")
        sys.exit(1)
    
    # Ask for confirmation before proceeding
    response = input("\nThis will create folders and download content for all artists. Continue? (y/n): ")
    if response.lower() in ['y', 'yes']:
        create_artist_folders_and_download()
        print("\n✓ Process completed!")
    else:
        print("Process cancelled.") 