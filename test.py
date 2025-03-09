from gradio_client import Client, handle_file

client = Client("https://2e1b4ade4618c88fb7.gradio.live/")
result = client.predict(
		image_input=handle_file('C:/Users/30309/Downloads/屏幕截图 2025-03-05 092443.png'),
		box_threshold=0.05,
		iou_threshold=0.1,
		use_paddleocr=True,
		imgsz=640,
		api_name="/process"
)
print(result)