from gradio_client import Client, file, handle_file
import os
outputDir = r"D:\Projects\VICTORIA"

client = Client("mouseland/cellpose")
result = client.predict(
        filepath=[handle_file(os.path.join(outputDir, "9-SE-1k-T1.bmp"))],
        resize=1000,
        max_iter=250,
        flow_threshold=0.4,
        cellprob_threshold=0,
        api_name="/cellpose_segment"
    )
print(result)
