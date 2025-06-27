import cv2
import pytesseract
from ultralytics import YOLO
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

print("Kamera açıldı, plaka algılaması başladı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, conf=0.5, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for x1,y1,x2,y2 in boxes:
        plate_img = frame[y1:y2, x1:x2]

        
        text = pytesseract.image_to_string(plate_img, config='--psm 7')
        text = ''.join([c for c in text if c.isalnum() or c==' ']).strip()
        if text:
            print("⏩ Plaka okundu:", text)
        else:
            print("⚠️ Plaka okunamadı")

        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Plaka Tespiti', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
