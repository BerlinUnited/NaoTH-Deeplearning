
import sys,requests,os
from vaapi.client import Vaapi
helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)
from pathlib import Path
from helper import get_file_from_server,label_dict

def get_log_server(timeout=2):
    url = "https://logs.berlin-united.com/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        print(f"Server {url} is alive.")
        return url
    except requests.exceptions.RequestException as e:
        print(f"Server {url} is not reachable: {e}")
        return "https://logs.naoth.de/"

def create_dataset_demo(client):
    response = client.annotations.list(id=168)
    log_server = get_log_server()
    for annotation in response:
        image = client.image.get(annotation.image)
        destination =  "./datasets/ball50/images/trainball50"
        filename = Path(image.image_url).name
        #get_file_from_server(log_server + image.image_url,f"{destination}/{filename}")
        
        label_dest = destination.replace("images","labels")
        with open(f"{label_dest}/{filename.split('.')[0]}.txt","w") as f:
            annotation = annotation.annotation["bbox"][0]
            label_id = label_dict.get(annotation["label"])
            
            # Get the bounding box coordinates
            x = annotation["x"] * 640  # Top-left x
            y = annotation["y"] * 480  # Top-left y
            width = annotation["width"]
            height = annotation["height"] 
            
            # Calculate the center coordinates
            cx = x + (640 / 2)  # Center x
            cy = y + (480 / 2)  # Center y
            
            # Normalize the coordinates
            cx = cx / 640
            cy = cy / 480

            f.write(f"{label_id} {cx} {cy} {width} {height}\n")


if __name__ == "__main__":
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    create_dataset_demo(client)