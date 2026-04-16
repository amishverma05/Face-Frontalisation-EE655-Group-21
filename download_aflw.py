import os, requests, zipfile, time

url = "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip"
file_path = "AFLW2000-3D.zip"

print(f"Downloading {file_path} from {url}...")
while True:
    headers = {}
    if os.path.exists(file_path):
        current_size = os.path.getsize(file_path)
        if current_size >= 200_000_000:  # Roughly expected size, stop looping if full
            break
        headers['Range'] = f'bytes={current_size}-'
        print(f"Resuming from {current_size} bytes...")
    
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=10)
        with open(file_path, "ab" if headers else "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: 
                    f.write(chunk)
                    print(f"Downloaded {os.path.getsize(file_path) / 1024 / 1024:.2f} MB", end='\r')
        if r.status_code in [200, 206] and int(r.headers.get('content-length', 0)) == 0:
            break
    except Exception as e:
        print(f"\nConnection dropped ({e}). Retrying in 5 seconds...")
        time.sleep(5)

print("\nDownload complete. Extracting...")
try:
    with zipfile.ZipFile(file_path, 'r') as z:
        z.extractall('data/')
    print("AFLW2000 dataset extracted successfully!")
except Exception as e:
    print(f"Failed to extract: {e}")
