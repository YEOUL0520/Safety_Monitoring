from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil

# 데이터셋 다운로드 (Roboflow)
rf = Roboflow(api_key="xkZo481q2JtZIdMBxgAM")
project = rf.workspace("knifedetection-cyv5m").project("helmet-evxi3-hlgs7")
version = project.version(1)
dataset = version.download("yolov11")

# yolov11n 모델 로드드
model = YOLO("yolov11n.pt")

# 학습습
results = model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=50,                # 우선 50으로 실행 후 진행 정도 보기로 함
    imgsz=640,
    batch=16,
    device="cuda",
    name="hardhat_yolov11",
    optimizer="AdamW",
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.001,
    flipud=0.1,
    mosaic=True,
    mixup=0.1
)

# 학습 결과 저장
weights_dir = model.trainer.save_dir + "/weights"
shutil.copy(os.path.join(weights_dir, "best.pt"), "custom_hardhat_v1.0.pt")
print("✅ 모델 저장 완료: custom_hardhat_v1.0.pt")

# TensorBoard 통해 실시간 손실/정확도 확인
print("\n📊 실시간 손실/정확도 보기:")
print(">> 실행: tensorboard --logdir runs/detect/")
print(">> 열기: http://localhost:6006")

# 학습 결과 확인
print("\n📈 Validation 평가:")
val_results = model.val()
print(f"📌 Precision     : {val_results.box.pr:.4f}")
print(f"📌 Recall        : {val_results.box.re:.4f}")
print(f"📌 mAP@0.5       : {val_results.box.map50:.4f}")
print(f"📌 mAP@0.5:0.95  : {val_results.box.map:.4f}")

print("\n📈 Test 평가:")
test_results = model.val(data=os.path.join(dataset.location, "data.yaml"), split="test")
print(f"📌 Precision     : {test_results.box.pr:.4f}")
print(f"📌 Recall        : {test_results.box.re:.4f}")
print(f"📌 mAP@0.5       : {test_results.box.map50:.4f}")
print(f"📌 mAP@0.5:0.95  : {test_results.box.map:.4f}")

# 결과 요약 
print("\n✅ Validation 결과:", val_results)
print("✅ Test 결과:", test_results)
