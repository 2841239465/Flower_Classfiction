"""
样题
"""

# 使用Flask搭建Web应用服务，完成图像的上传和识别
import os
from flask import Flask, render_template, request, flash, make_response, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from flask import send_file
from inference import infer
# from engine.util import send_link
UPLOAD_FOLDER = "/images"

app = Flask(__name__,template_folder='static/templates')

# 允许上传type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "PNG", "JPG", 'JPEG'])

# 检查上传文件类型
def allowed_file(filename):
    return '.' in filename and filename.split('.', 1)[1] in ALLOWED_EXTENSIONS

# route_prefix = os.environ["JUPYTERHUB_SERVICE_PREFIX"] + "export/"
# print("点击下方链接进入demo页面：")
# send_link(route_prefix + "index" )

# 实现图片上传的视图函数upload_file，做到能成功渲染上传图片的模板文件upload.html。
# 在图像上传视图函数upload_file中加入判断语句，使得上传的图片能保存在平台中的个人空间目录下（/space)。
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 获取上传文件
        file = request.files['file']
        # 检查文件对象是否存在且合法
        if file and allowed_file(file.filename):  # 哪里规定file都有什么属性
            filename = secure_filename(file.filename)  # 把汉字文件名抹掉了，所以下面多一道检查
            print(filename)
            if filename != file.filename:
                flash("only support ASCII name")
                return render_template('upload.html')
            # save
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))  # 现在似乎不会出现重复上传同名文件的问题
            except FileNotFoundError:
                os.makedirs(UPLOAD_FOLDER)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('update', fileName=filename))
        else:
            return 'Upload Failed'
    else:  # GET方法
        return render_template('upload.html')
#

# 调用infer函数对上传的图像进行分类并保存预测的结果
@app.route('/uploads/<path:fileName>', methods=['POST', 'GET'])
def update(fileName):
    """输入url加载图片，并返回预测值；上传图片，也会重定向到这里"""
    result = render_photo_as_page(fileName)
    return render_template('show.html', fname=fileName, result=result)
#
# 将infer替换成scene_recogintion对上传的图像进行分类并保存预测的结果
def render_photo_as_page(filename):
    """请在此作答"""
    # 上传文件夹和static分离
    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
    # 这里要求jpg还是png都必须保存成png，因为html里是写死的
    img.save(os.path.join(UPLOAD_FOLDER, filename))
    # 实现推理计算
    preds = infer(os.path.join(UPLOAD_FOLDER, filename))
    # 步骤6，实现场景分类
    # from application import scene_recognition
    # preds = scene_recognition(os.path.join(UPLOAD_FOLDER, filename))
    result = {}
    result["prediction"] = preds
    result["fileName"] = filename
    return result
## 网页获得上传的图像文件
@app.route('/get/<path:fileName>', methods=['GET'])
def get_file(fileName):
    file_path = os.path.join(UPLOAD_FOLDER, fileName)
    return send_file(file_path)

## 实现反馈视图函数thanks，做到能处理show.html中form表单提交的预测结果的功能，以及成功渲染反馈的模板文件thanks.html。
@app.route('/thanks', methods=['POST', 'GET'])
def thanks():
    category = request.form["Correctness"]  # True or False
    prediction = request.form['prediction']
    fileName = request.form['filename']
    print(category, prediction, fileName)
    # 调用update去更新模型
    return render_template('thanks.html', category=category, prediction=prediction, fileName=fileName)


if __name__ == '__main__':
    app.run()
    # 运行之后将输出的route_prefix+“index”作为url，打开浏览器输入这个url即可访问网络服务
