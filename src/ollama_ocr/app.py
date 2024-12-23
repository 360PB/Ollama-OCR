import streamlit as st
#from .ocr_processor import OCRProcessor
import tempfile
import os
import sys
from PIL import Image
import json
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from ocr_processor import OCRProcessor

# 页面配置
st.set_page_config(
    page_title="OCR 实验室",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    .gallery-item {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        background: white;
    }
    </style>
    """, unsafe_allow_html=True)

def get_available_models():
    return ["llava:7b", "llama3.2-vision:11b"]

def process_single_image(processor, image_path, format_type, enable_preprocessing):
    """处理单张图片并返回结果"""
    try:
        result = processor.process_image(
            image_path=image_path,
            format_type=format_type,
            preprocess=enable_preprocessing
        )
        return result
    except Exception as e:
        return f"处理图片时出错: {str(e)}"

def process_batch_images(processor, image_paths, format_type, enable_preprocessing):
    """处理多张图片并返回结果"""
    try:
        results = processor.process_batch(
            input_path=image_paths,
            format_type=format_type,
            preprocess=enable_preprocessing
        )
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("🔍 OCR 实验室")
    st.markdown("<p style='text-align: center; color: #666;'>由 Ollama 视觉模型提供支持</p>", unsafe_allow_html=True)

    # 侧边栏控件
    with st.sidebar:
        st.header("🎮 控制面板")
        
        selected_model = st.selectbox(
            "🤖 选择视觉模型",
            get_available_models(),
            index=0,
        )
        
        format_type = st.selectbox(
            "📄 输出格式",
            ["markdown", "text", "json", "structured", "key_value"],
            help="选择提取文本的输出格式"
        )

        max_workers = st.slider(
            "🔄 并行处理数量",
            min_value=1,
            max_value=8,
            value=2,
            help="批量处理时的并行图片处理数量"
        )

        enable_preprocessing = st.checkbox(
            "🔍 启用预处理",
            value=True,
            help="应用图像增强和预处理"
        )
        
        st.markdown("---")
        
        # 模型信息框
        if selected_model == "llava:7b":
            st.info("LLaVA 7B: 高效的视觉语言模型，适用于实时处理")
        else:
            st.info("Llama 3.2 Vision: 高精度模型，适用于复杂文本提取")

    # 初始化 OCR 处理器
    processor = OCRProcessor(model_name=selected_model, max_workers=max_workers)

    # 主内容区域，包含标签页
    tab1, tab2 = st.tabs(["📸 图片处理", "ℹ️ 关于"])
    
    with tab1:
        # 文件上传区域，支持多文件
        uploaded_files = st.file_uploader(
            "将图片拖放到此处",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="支持的格式: PNG, JPG, JPEG, TIFF, BMP, PDF"
        )

        if uploaded_files:
            # 创建临时目录保存上传的文件
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                
                # 保存上传的文件并收集路径
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    image_paths.append(temp_path)

                # 在画廊中显示图片
                st.subheader(f"📸 输入图片 ({len(uploaded_files)} 个文件)")
                cols = st.columns(min(len(uploaded_files), 4))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(uploaded_file)
                        st.image(image, use_container_width=True, caption=uploaded_file.name)

                # 处理按钮
                if st.button("🚀 开始处理图片"):
                    with st.spinner("正在处理图片..."):
                        if len(image_paths) == 1:
                            # 单张图片处理
                            result = process_single_image(
                                processor, 
                                image_paths[0], 
                                format_type,
                                enable_preprocessing
                            )
                            st.subheader("📝 提取的文本")
                            st.markdown(result)
                            
                            # 下载按钮
                            st.download_button(
                                "📥 下载结果",
                                result,
                                file_name=f"ocr_result.{format_type}",
                                mime="text/plain"
                            )
                        else:
                            # 批量处理
                            results = process_batch_images(
                                processor,
                                image_paths,
                                format_type,
                                enable_preprocessing
                            )
                            
                            # 显示统计信息
                            st.subheader("📊 处理统计信息")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("总图片数", results['statistics']['total'])
                            with col2:
                                st.metric("成功数量", results['statistics']['successful'])
                            with col3:
                                st.metric("失败数量", results['statistics']['failed'])

                            # 显示结果
                            st.subheader("📝 提取的文本")
                            for file_path, text in results['results'].items():
                                with st.expander(f"结果: {os.path.basename(file_path)}"):
                                    st.markdown(text)

                            # 显示错误信息
                            if results['errors']:
                                st.error("⚠️ 部分文件处理失败:")
                                for file_path, error in results['errors'].items():
                                    st.warning(f"{os.path.basename(file_path)}: {error}")

                            # 下载所有结果为 JSON
                            if st.button("📥 下载所有结果"):
                                json_results = json.dumps(results, indent=2)
                                st.download_button(
                                    "📥 下载结果 JSON",
                                    json_results,
                                    file_name="ocr_results.json",
                                    mime="application/json"
                                )

    with tab2:
        st.header("关于 OCR 实验室")
        st.markdown("""
        本应用使用最先进的 Ollama 视觉语言模型从图片中提取文本。
        
        ### 功能特点:
        - 🖼️ 支持多种图片格式
        - 📦 批量处理能力
        - 🔄 并行处理
        - 🔍 图像预处理和增强
        - 📊 多种输出格式
        - 📥 结果轻松下载
        
        ### 模型:
        - **LLaVA 7B**: 高效的视觉语言模型，适用于实时处理
        - **Llama 3.2 Vision**: 高精度模型，适用于复杂文档
        """)

if __name__ == "__main__":
    main()
