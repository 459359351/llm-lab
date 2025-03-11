import fitz
import os
import base64
import requests
import time
from PIL import Image
from io import BytesIO
import json
from openai import OpenAI


def extract_images(pdf_path, output_dir):
    """提取PDF图片并返回图片路径列表"""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_ext = base_image["ext"]
                    filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(base_image["image"])
                    image_paths.append(filepath)
        return image_paths
    except Exception as e:
        print(f"图片提取失败：{str(e)}")
        return []


def image_to_base64(image_path):
    """图片转Base64编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_table(image_base64):
    """调用MiniCPM-V检测表格"""
    prompt = (
        "请严格按以下步骤处理：\n"
        "1. 判断图片是否包含表格，若无则直接返回'无表格'\n"
        "2. 若有表格，提取完整表格内容\n"
        "3. 转换为格式正确的Markdown表格（确保列对齐）"
    )

    try:
        """发送请求到Ollama接口"""
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": "minicpm-v:latest",
            "prompt": "请识别图片中是否有表格，如果有则提取,并使用markdown格式展示，如果没有则略过，返回无",
            "stream": False,
            "images": [image_base64]
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"模型调用失败：{str(e)}")
        return ""


def qwen_api(image_base64, question):
    # 将xxxx/test.png替换为你本地图像的绝对路径
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-92ba8ab9e3ba418f91505b0bdec8354d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一名图片识别内容提取助手."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)

def process_pdf(pdf_path, output_dir):
    """主处理流程"""
    # 提取图片
    images = extract_images(pdf_path, output_dir)
    print(f"成功提取 {len(images)} 张图片")

    # 处理每张图片
    for img_path in images:
        print(f"\n处理图片：{os.path.basename(img_path)}")
        start_time = time.time()

        # 转换为Base64
        base64_str = image_to_base64(img_path)

        # 调用本地模型
        # result = detect_table(base64_str)

        question = "请严格按以下步骤处理：1. 判断图片是否包含表格，若无则直接返回'无表格'；2. 若有表格，提取完整表格内容；3. 转换为格式正确的Markdown表格（确保列对齐）"

        #调用线上api
        result = qwen_api(base64_str, question)

        print(f"处理耗时：{time.time() - start_time:.2f}秒")
        print(result)


if __name__ == "__main__":
    pdf_file = "pdf_with_table.pdf"  # PDF文件路径
    output_folder = "output_folder"  # 图片输出目录

    process_pdf(pdf_file, output_folder)