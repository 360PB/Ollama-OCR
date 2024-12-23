import subprocess

def start_ollama_service(model_name="llama3.2-vision:11b"):
    """
    启动 Ollama 服务并加载指定模型。
    
    Args:
        model_name (str): 要加载的模型名称。
    """
    print("正在启动 Ollama 服务...")
    # 启动 Ollama 服务
    serve_command = "ollama serve"
    serve_process = subprocess.Popen(serve_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待服务启动（这里可以调整等待时间，根据实际情况）
    time.sleep(3)
    
    print(f"正在加载模型: {model_name}...")
    # 运行模型
    run_command = f"ollama run {model_name}"
    run_process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待模型加载（这里可以调整等待时间，根据实际情况）
    time.sleep(5)
    
    # 检查模型是否成功加载
    print("检查模型是否加载成功...")
    try:
        # 使用GET /api/ps API来检查模型是否加载
        response = requests.get("http://localhost:11434/api/ps")
        response.raise_for_status()  # 如果响应状态码不是200，将抛出HTTPError异常
        models = response.json().get("models", [])
        if any(model["name"] == model_name for model in models):
            print(f"模型 {model_name} 已成功加载。")
        else:
            print(f"模型 {model_name} 加载失败。")
    except requests.RequestException as e:
        print(f"检查模型加载状态时出错: {e}")

if __name__ == "__main__":
    import time
    import requests
    start_ollama_service(model_name="llama3.2-vision:11b")