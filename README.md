# Lab 3: Machine Learning สำหรับการจำแนกการใช้ที่ดิน

ลิงก์ code ใน colab https://colab.research.google.com/drive/1SKcclUPygEGDJiqNvWblgbkLWeKkiTJs?usp=sharing

* **ออกแบบ Training Strategy**

  ▸ ใช้ประเภทที่ดิน 5 class ได้แก่
Water (แหล่งน้ำ), Agriculture (เกษตรกรรม) ,Forest (ป่าไม้), Built-up (สิ่งปลูกสร้าง) และ Bareland (พื้นดินโล่ง)

  ▸ หา Training Samples โดยการวาด Polygon เองผ่าน Google Earth Engine Code Editor

  ▸ จะแบ่ง Train/Validation เป็นแบบแบ่งสุ่ม 80/20

  ▸ Feature ที่จะใช้มี 
  Red, Green, Blue, NIR, SWIR, NDVI, NDWI, SAVI เนื่องจาก

      Red, Green, Blue เป็น feature พื้นฐาน
  
      NIR, SWIR ใช้แยกความชื้น, น้ำ
  
      NDVI ใช้แยก vegetation ได้ดี
  
      NDWI ใช้แยกแหล่งน้ำได้ดี
  
      SAVI ใช้แยก vegetation ที่ปนกับ Bareland ได้ดีขึ้น

* **ผลการเปรียบเทียบ**

โดยภาพรวม RF และ XGB สามารถจำแนกการใช้ประโยชน์ที่ดินได้ดี แต่ XGB ให้ performance ที่เสถียรกว่าในหลาย class

Class Water RF มี PA,UA อยู่ที่ 0.92 ส่วน XGB PA,UA อยู่ที่ 1.00 และ 0.87 ตามลำดับ แสดงให้เห็นว่า XGB จับ Water ได้หมด แต่มี false positive ที่โมเดลสับสนระหว่าง water กับ agriculture

Class Agriculture ทั้ง 2 โมเดลมีค่า PA, UA อยู่ที่ 0.57 และ 1.00 ตามลำดับ แสดงให้เห็นถึงความ conservative ของโมเดลที่มักทำนายโดยใช้ค่าค่าเฉลี่ย (Mean/Median)

Class Forest RF มี PA,UA อยู่ที่ 1.00 และ 0.83 ส่วน XGB PA,UA อยู่ที่ 1.00 แสดงให้เห็นว่า Forest มี spectral signature ที่ชัดเจนทำให้ทั้ง 2 โมเดลทำนายได้ดีเยี่ยม โดยเฉพาะ XGB ที่ทำนายถูกต้องทั้งหมด

Class Buit-up RF มี PA อยู่ที่ 0.73 ส่วน XGB PA อยู่ที่ 0.82 แสดงให้เห็นว่า XGB ทำนายได้ดีกว่าแต่ก็ยังมี confusion อยู่

Class Bareland RF มี UA อยู่ที่ 0.69 ส่วน XGB UA อยู่ที่ 0.75 แสดงให้เห็นว่ามี false positive เยอะ มีความสับสนกับ Agriculture และ Built-up

* **คำถาม**

**1. ถ้าเพิ่ม Training Samples อีก 2 เท่า ความแม่นยำจะเพิ่มขึ้นเท่าไหร่? ทดสอบและอธิบายผล**

  เมื่อเพิ่ม Training Samples อีก 2 เท่า ความแม่นยำในเชิง Overall Accuracy RF จากเดิม 0.86 เพิ่มขึ้นเป็น 0.90 ในขณะที่ XGBoost มี OA ลดลงเล็กน้อยจากเดิม 0.90 เป็น 0.89

**2. Spatial Autocorrelation ในข้อมูล Training มีผลต่อความแม่นยำที่รายงานอย่างไร?**

   จากการทดลอง Run โค้ดเพิ่มเติม Spatial Autocorrelation ในข้อมูล Training มีผลต่อความแม่นยำสูงกว่า แบบแบ่ง 80/20 แล้วสุ่มเล็กน้อย (0.86 กับ 0.88)

**3. Class ใดที่โมเดลทำได้แย่ที่สุด — แก้ได้ด้วยวิธีใดบ้าง?**

  Class Agriculture โมเดล RF ทำได้แย่สุด ถึงแม้จะมีค่า UA = 1.00 แต่ PA,F1-score มีค่า 0.57 และ 0.72 ตามลำดับ ซึ่งอาจแก้ได้ด้วยการเพิ่ม training samples, ใช้ NDVI time-series ที่ช่วยให้สามารถแยกนาข้าวได้ดีขึ้น หรือจะแก้ด้วยการปรับ class definition แยกเป็นนาข้าวกับเกษตรกรรมอื่น ๆ หรือพื้นที่เกษตรแบบเปียกกับพื้นที่เกษตรแบบแห้ง

**4. ถ้าต้องทำซ้ำ Lab นี้สำหรับพื้นที่อื่น อะไรคือสิ่งที่ต้องเปลี่ยน และอะไรที่ใช้ซ้ำได้?**

  ถ้าจะเปลี่ยนพื้นที่ศึกษาสิ่งที่ใช้ซ้ำได้ คือ ขั้นตอนการรันโค้ด (Code pipeline) ส่วน Class definition,Feature ที่ใช้, Model, Evalution metrics อาจจะใช้ซ้ำหรือไม่ใช้ก็ได้ขึ้นอยู่กับบริบทของพื้นที่ แต่สิ่งที่ต้องเปลี่ยนคือ training samples ต้องทำการสุ่มใหม่ทุกพื้นที่
