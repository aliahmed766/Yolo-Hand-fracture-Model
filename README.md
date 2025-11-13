# Hand Fracture Detection using YOLOv8 + Streamlit

An AI-powered web app that detects **hand bone fractures** from X-ray images using a trained **YOLOv8** model.  
Built with **Streamlit**, **PyTorch**, and **Ultralytics YOLOv8**, this app provides a simple interface for uploading X-ray images and visualizing fracture detection results.

---

## 🚀 Features

- 🧠 Detects fractures in hand X-rays using YOLOv8  
- 📤 Upload `.jpg`, `.jpeg`, or `.png` files directly  
- ⚡ Fast, real-time inference  
- 🖼️ Displays detection results with bounding boxes  
- ☁️ 100% compatible with **Streamlit Cloud**

---

## 🗂️ Project Structure
Yolo-Hand-fracture-Model/
│
├── app.py # Streamlit web app
├── best.pt # YOLOv8 trained weights (your model)
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── .gitattributes # Git config


---

## 💻 Installation & Local Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aliahmed766/Yolo-Hand-fracture-Model.git
cd Yolo-Hand-fracture-Model

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


Then open the local URL shown in your terminal (usually http://localhost:8501
).

⚙️ Requirements

The requirements.txt file contains:

streamlit
ultralytics
torch
torchvision
numpy
Pillow


⚠️ Note: We intentionally removed opencv-python because it causes import errors on Streamlit Cloud (Python 3.13).
The app works perfectly without it.

🧩 Model Information

Model: YOLOv8 custom-trained on hand X-ray images

File: best.pt (included in repo root)

Framework: PyTorch + Ultralytics YOLOv8

Task: Object Detection (Fracture Localization)

☁️ Deploy on Streamlit Cloud

Go to Streamlit Cloud

Click New app

Select your GitHub repo → aliahmed766/Yolo-Hand-fracture-Model

Set Main file path: app.py

Deploy 🎉

Your app will be live and accessible online instantly!

📸 Example Output
Uploaded X-ray	Detection Result

	

The app overlays a bounding box around the detected fracture area.

🔧 Troubleshooting

❌ cv2 ImportError:
Remove opencv-python or opencv-contrib-python from requirements.txt.

❌ Model Load Warning (weights_only=True):
This is a PyTorch 2.6+ safety warning. Your model is fine as long as it’s from a trusted source.

❌ Streamlit Error:
Run:

pip install --upgrade streamlit ultralytics torch torchvision Pillow numpy

👨‍💻 Author

Ali Ahmed
🎓 B.S. Software Engineering | AI & ML Enthusiast

