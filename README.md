# Brain Tumor Detection

مشروع بسيط لتصنيف صور MRI إلى فئتين: `no` (بدون ورم) و`yes` (وجود ورم) باستخدام نموذج CNN مبني على TensorFlow/Keras.

## التثبيت

يفضل استخدام بيئة افتراضية ثم تثبيت الاعتماديات:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## تدريب النموذج

يوجد أرشيف البيانات `brain_tumor_datase.zip` داخل المشروع. يستخرج `mainTrain.py` الأرشيف تلقائيًا عند الحاجة، ثم يبحث عن مجلدي `no` و`yes`:

```bash
python mainTrain.py
```

لتغيير عدد العصور أو مكان حفظ النموذج:

```bash
python mainTrain.py --epochs 10 --output braintumor10Epoccategorical.h5
```

## اختبار صورة واحدة

مرر مسار صورة MRI بدل المسار الثابت القديم الخاص بجهاز Windows:

```bash
python mainTest.py path/to/image.jpg
```

ويمكن تحديد نموذج مختلف:

```bash
python mainTest.py path/to/image.jpg --model path/to/model.h5
```

## تشغيل الواجهة الرسومية

بعد تثبيت TensorFlow وTkinter وتشغيل النموذج:

```bash
python gui_test.py
```

ثم اختر صورة من زر **Choose**. الواجهة تستخدم نفس تجهيز الصور والتطبيع المستخدمين في التدريب والاختبار.

> نتيجة النموذج إرشادية وليست تشخيصًا طبيًا. لا ينبغي استخدامها بدل تقييم طبيب مختص.
