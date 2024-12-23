import os
import base64
import requests
import json
import cv2
import tempfile
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Optional, Dict, Any, List, Union


class OCRProcessor:
    def __init__(self, model_name: str = "llama3.2-vision:11b", max_workers: int = 1):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"
        self.max_workers = max_workers

    def _convert_to_short_path(self, long_path: str) -> str:
        """
        将长路径（可能包含中文）转换为短路径（Windows 下的 8.3 格式路径）。
        """
        if os.name == 'nt':  # 仅在 Windows 下适用
            import ctypes
            buffer = ctypes.create_unicode_buffer(260)
            ctypes.windll.kernel32.GetShortPathNameW(long_path, buffer, 260)
            return buffer.value
        return long_path  # 非 Windows 系统直接返回原路径

    def _read_image(self, image_path: str) -> np.ndarray:
        """
        安全读取图片，支持中文路���。
        """
        try:
            # 转换为短路径（解决中文路径问题）
            safe_path = self._convert_to_short_path(image_path)
            with open(safe_path, "rb") as f:
                file_bytes = f.read()
                image_array = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图片文件: {image_path}")
            return image
        except Exception as e:
            raise RuntimeError(f"读取图片失败: {image_path}, 错误: {e}")

    def _encode_image(self, image_path: str) -> str:
        """将图片转换为 Base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _preprocess_image(self, image_path: str) -> str:
        """
        预处理图片：
        - 将 PDF 转换为图片（如果需要）
        - 自动旋转
        - 增强对比度
        - 降噪
        """
        try:
            # 处理 PDF 文件
            if image_path.lower().endswith('.pdf'):
                pages = convert_from_path(image_path)
                if not pages:
                    raise ValueError("无法将 PDF 转换为图片")
                # 保存第一页为临时图片
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    temp_path = temp_file.name
                pages[0].save(temp_path, 'JPEG')
                image_path = temp_path

            # 读取图片
            image = self._read_image(image_path)

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用 CLAHE 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 降噪
            denoised = cv2.fastNlMeansDenoising(enhanced)

            # 保存预处理后的图片
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                preprocessed_path = temp_file.name
            cv2.imwrite(preprocessed_path, denoised)

            return preprocessed_path
        except Exception as e:
            raise RuntimeError(f"预处理图片时出错: {image_path}, 错误: {e}")

    def process_image(self, image_path: str, format_type: str = "markdown", preprocess: bool = True) -> str:
        """
        处理单张图片并提取文本
        """
        try:
            if preprocess:
                image_path = self._preprocess_image(image_path)

            image_base64 = self._encode_image(image_path)

            # 清理临时文件
            if image_path.endswith(('_preprocessed.jpg', '_temp.jpg')):
                os.remove(image_path)

            # 提示模板
            prompts = {
                "markdown": """请分析提供的图片内容，并以 Markdown 格式输出提取的文本，要求如下：
                - 使用简体中文输出所有文本内容。
                - 根据图片中的视觉层级结构，使用标题（#、##、###）表示不同层级的标题和章节。
                - 使用列表符号（-）表示列表，并保持原有的列表结构。
                - 对需要强调的内容使用 Markdown 格式（如加粗、斜体等）。
                - 如果图片中包含表格，请使用 Markdown 表格格式表示。
                - 尽可能保留图片中的文本层级、布局和格式。
                - 确保所有可见文本都被提取，包括部分被遮挡或旋转的文本。""",

                "text": """请从提供的图片中提取所有可见文本，并以纯文本格式输出，要求如下：
                - 使用简体中文输出所有文本内容。
                - 尽可能保持原始的文本布局和换行格式。
                - 包括所有可见文本，即使部分文本被遮挡、旋转或使用非标准字体。
                - 不需要任何格式化或元数据，仅输出纯文本内容。
                - 确保提取的文本清晰、准确，避免 OCR 错误。""",

                "json": """请从提供的图片中提取所有可见文本，并以 JSON 格式输出，要求如下：
                - 使用简体中文输出所有文本内容。
                - 将文本分组为逻辑部分（如标题、正文、页脚等），并使用描述性键名（如 "标题"、"正文"、"段落"、"表格"、"列表"）表示。
                - 保留内容的层级结构，反映图片中的视觉布局。
                - 如果图片中包含表格，请将其表示为嵌套数组，按行和列组织数据。
                - 确保 JSON 格式正确且缩进清晰，便于阅读。
                - 包括所有可见文本，即使部分文本被遮挡或旋转。
                - 如果图片中包含无法识别的内容，请标注为 `"无法识别"`。
                - 示例输出格式：
                ```json
                {
                  "标题": "这是标题",
                  "正文": "这是正文内容",
                  "表格": [
                    ["列1", "列2", "列3"],
                    ["数据1", "数据2", "数据3"]
                  ],
                  "列表": ["项目1", "项目2", "项目3"]
                }
                """,

                "structured": """请从提供的图片中提取所有可见文本，并以结构化格式输出，要求如下：
                - 使用简体中文输出所有文本内容。
                - 识别并格式化表格，保留表格中的行和列结构。
                - 提取有序或无序列表，并保持其嵌套层级。
                - 保留图片中的层级关系（如标题、子标题、段落等），并使用清晰的标签或缩进表示。
                - 包括所有可见文本，即使部分文本被遮挡或旋转。
                - 确保输出内容清晰、有条理，并尽量反映原始图片的视觉结构。
                - 如果图片中包含无法识别的内容，请标注为 `[无法识别]`。
                - 示例输出格式：
                标题: 这是标题
                正文: 这是正文内容
                表格:

                行1: 列1, 列2, 列3
                行2: 数据1, 数据2, 数据3
                列表:
                项目1
                项目2
                项目3
                """,

                "key_value": """请从提供的图片中提取所有以键值对形式出现的文本，并以键值对格式输出，要求如下：
                - 使用简体中文输出所有文本内容。
                - 识别标签（键）及其对应的值，即使它们在视觉上分离。
                - 提取表单字段及其内容，保留字段名称与值之间的对应关系。
                - 每个键值对以 "键: 值" 的格式输出，每行一个键值对。
                - 如果一个键对应多个值，请将值以逗号分隔的形式输出。
                - 包括所有可见的键值对，即使部分内容被遮挡或旋转。
                - 如果图片中包含无法识别的内容，请标注为 `[无法识别]`。
                - 示例输出格式：
                """
                }

            # 获取对应的提示
            prompt = prompts.get(format_type, prompts["text"])

            # 准备请求数据
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [image_base64]
            }

            # 调用 Ollama API
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()

            result = response.json().get("response", "")

            # 如果是 JSON 格式，尝试解析
            if format_type == "json":
                try:
                    json_data = json.loads(result)
                    return json.dumps(json_data, indent=2)
                except json.JSONDecodeError:
                    return result

            return result
        except Exception as e:
            return f"处理图片时出错: {str(e)}"

    def process_batch(
        self,
        input_path: Union[str, List[str]],
        format_type: str = "markdown",
        recursive: bool = False,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        批量处理多张图片
        """
        # 收集所有图片路径
        image_paths = []
        if isinstance(input_path, str):
            base_path = Path(input_path)
            if base_path.is_dir():
                pattern = '**/*' if recursive else '*'
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.tiff']:
                    image_paths.extend(base_path.glob(f'{pattern}{ext}'))
            else:
                image_paths = [base_path]
        else:
            image_paths = [Path(p) for p in input_path]

        results = {}
        errors = {}

        # 并行处理图片
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image, str(path), format_type, preprocess): path
                    for path in image_paths
                }

                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[str(path)] = future.result()
                    except Exception as e:
                        errors[str(path)] = str(e)
                    pbar.update(1)

        return {
            "results": results,
            "errors": errors,
            "statistics": {
                "total": len(image_paths),
                "successful": len(results),
                "failed": len(errors)
            }
        }
