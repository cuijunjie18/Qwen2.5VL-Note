# Qwen2.5VL本地部署

## 目录

[环境配置](#环境配置)  
[模型下载](#模型下载)
[推理测试](#demo测试)

## 环境配置

### 使用conda配置
- conda创建独立虚拟环境
  ```shell
  conda create --name Qwen2.5VL python=3.10
  # 注意conda要换源，否则pip安装很慢
  # conda info可以查看换源情况
  ```
  <br>
- clone本体github项目
  ```shell
  git clone https://github.com/QwenLM/Qwen2.5-VL.git
  ```
  <br>
- 安装依赖
  ```shell
  pip install transformers==4.51.3 accelerate
  pip install qwen-vl-utils
  pip install -r requirements_web_demo.txt
  pip install decord
  pip install flash_attn
  ```

  其中requirements_web_demo.txt内容如下
  ```shell
  # Core dependencies
  gradio==5.4.0
  gradio_client==1.4.2
  qwen-vl-utils==0.0.10
  transformers-stream-generator==0.0.4
  torch==2.4.0
  torchvision==0.19.0
  # git+https://github.com/huggingface/transformers.git
  accelerate
  av

  # Optional dependency
  # Uncomment the following line if you need flash-attn
  # flash-attn==2.6.1 # 这个版本已经不对了
  ```
  注意要注释掉
  ```shell
  git+https://github.com/huggingface/transformers.git
  ```
  **因为huggingface大概率需要挂梯子**
  <br>

- 检查最终完整的依赖

  ```shell
  pip freeze > pip_depend.txt
  ```
  最后得到的pip_depend.txt如下

  ```txt
  accelerate==1.7.0
  aiofiles==23.2.1
  annotated-types==0.7.0
  anyio==4.9.0
  av==14.4.0
  certifi==2025.4.26
  charset-normalizer==3.4.2
  click==8.1.8
  decord==0.6.0
  einops==0.8.1
  exceptiongroup==1.3.0
  fastapi==0.115.12
  ffmpy==0.5.0
  filelock==3.18.0
  flash_attn==2.7.4.post1
  fsspec==2025.5.0
  gradio==5.4.0
  gradio_client==1.4.2
  h11==0.16.0
  httpcore==1.0.9
  httpx==0.28.1
  huggingface-hub==0.31.4
  idna==3.10
  Jinja2==3.1.6
  markdown-it-py==3.0.0
  MarkupSafe==2.1.5
  mdurl==0.1.2
  mpmath==1.3.0
  networkx==3.4.2
  numpy==2.2.6
  nvidia-cublas-cu12==12.1.3.1
  nvidia-cuda-cupti-cu12==12.1.105
  nvidia-cuda-nvrtc-cu12==12.1.105
  nvidia-cuda-runtime-cu12==12.1.105
  nvidia-cudnn-cu12==9.1.0.70
  nvidia-cufft-cu12==11.0.2.54
  nvidia-cufile-cu12==1.11.1.6
  nvidia-curand-cu12==10.3.2.106
  nvidia-cusolver-cu12==11.4.5.107
  nvidia-cusparse-cu12==12.1.0.106
  nvidia-cusparselt-cu12==0.6.3
  nvidia-nccl-cu12==2.20.5
  nvidia-nvjitlink-cu12==12.6.85
  nvidia-nvtx-cu12==12.1.105
  orjson==3.10.18
  packaging==25.0
  pandas==2.2.3
  pillow==11.2.1
  psutil==7.0.0
  pydantic==2.11.4
  pydantic_core==2.33.2
  pydub==0.25.1
  Pygments==2.19.1
  python-dateutil==2.9.0.post0
  python-multipart==0.0.12
  pytz==2025.2
  PyYAML==6.0.2
  qwen-vl-utils==0.0.10
  regex==2024.11.6
  requests==2.32.3
  rich==14.0.0
  ruff==0.11.10
  safehttpx==0.1.6
  safetensors==0.5.3
  semantic-version==2.10.0
  shellingham==1.5.4
  six==1.17.0
  sniffio==1.3.1
  starlette==0.46.2
  sympy==1.14.0
  tokenizers==0.21.1
  tomlkit==0.12.0
  torch==2.4.0
  torchvision==0.19.0
  tqdm==4.67.1
  transformers==4.51.3
  transformers-stream-generator==0.0.4
  triton==3.0.0
  typer==0.15.4
  typing-inspection==0.4.1
  typing_extensions==4.13.2
  tzdata==2025.2
  urllib3==2.4.0
  uvicorn==0.34.2
  websockets==12.0
  ```
  <br>

### 使用uv配置

- 初始化uv环境
  ```shell
  mkdir Qwen2.5VL
  cd Qwen2.5VL
  uv init
  uv python pin 3.11
  ```
  测试初始环境
  ```shell
  uv run main.py
  ```

- 列出依赖表
  ```shell
  vim requirements.txt
  ```
  ```vim
  # 必须
  torch
  torchvision
  transformers
  accelerate
  qwen-vl-utils
  decord  # 视频处理

  # 我的工具
  tqdm
  opencv-python
  torchinfo
  matplotlib
  ```

- 安装基础环境
  ```shell
  uv add -r requirements.txt
  ```

- flash-attn安装(~~未解决~~已解决)
  通常来说，直接pip安装，或者uv add安装容易出现**依赖错误或者版本错误**，使用官网包安装.

  在官网找到对应的cuda、torch、cpython版本和abiTrue or False,这个就自己两个都试一下，能跑通就行.
  官网版本：https://github.com/Dao-AILab/flash-attention/releases

  ```shell
  uv add xxxx.whl
  ```

## 模型下载

- 安装模型本体
  魔搭社区(huggingface也行):https://modelscope.cn/collections/Qwen25-VL-58fbb5d31f1d47
  找到相应的硬件带得动的模型大小.

  这里我用git安装(**缺点：无可视化进度、会把log也clone下来**)，魔搭社区也有其他安装方法.
  ```shell
  git clone https://www.modelscope.cn/Qwen/Qwen2.5-VL-7B-Instruct.git
  ```

  使用modelscope下载模型
  ```shell
  pip install modelscope
  modelscope download --model Qwen/Qwen2.5-VL-32B-Instruct --local_dir ./dir # 指定本地路径
  ```
  <br>
- 链接模型文件到Qwen2.5Vl本体(option/可选)
  ```shell
  cd Qwen2.5VL
  ln -s <real_model_path> models
  ```
  然后可以在Qwen2.5本体处调用模型了，较为方便.

## demo测试

- 图片理解

  ```py
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
  from qwen_vl_utils import process_vision_info
  
  # default: Load the model on the available device(s)
  model = Qwen2_5_VLForConditionalGeneration.  from_pretrained(
      "models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto",   device_map="auto"
  ) # 注意更改模型路径
  
  # We recommend enabling flash_attention_2 for better   acceleration and memory saving, especially in   multi-image and video scenarios.
  # model = Qwen2_5_VLForConditionalGeneration.  from_pretrained(
  #     "Qwen/Qwen2.5-VL-7B-Instruct",
  #     torch_dtype=torch.bfloat16,
  #     attn_implementation="flash_attention_2",
  #     device_map="auto",
  # )
  
  # default processor
  processor = AutoProcessor.from_pretrained("models/Qwen2.  5-VL-7B-Instruct") # 注意更改模型路径
  
  # The default range for the number of visual tokens per   image in the model is 4-16384.
  # You can set min_pixels and max_pixels according to   your needs, such as a token range of 256-1280, to   balance performance and cost.
  # min_pixels = 256*28*28
  # max_pixels = 1280*28*28
  # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.  5-VL-7B-Instruct", min_pixels=min_pixels,   max_pixels=max_pixels)
  
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": "images/view.jpg", # 图片自己设置
              },
              {"type": "text", "text": "描述这张图片"},
          ],
      }
  ]
  
  # Preparation for inference
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  image_inputs, video_inputs = process_vision_info  (messages)
  inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
  )
  inputs = inputs.to(model.device)
  
  # Inference: Generation of the output
  generated_ids = model.generate(**inputs,   max_new_tokens=128)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip  (inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True,   clean_up_tokenization_spaces=False
  )
  print(output_text)
  ```
  运行
  ```py
  python3 demo.py
  ```
  正确输出推理内容即可.
  <br>

- 视频推理

  ```py
  # 修改对应的messages即可
  # video message
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "video",
                  "video": "videos/  100325-655758396_resize1080p.mp4",
                  "max_pixels": 360 * 420,
                  "fps": 1.0,
              },
              {"type": "text", "text": "描述这段视频"}
          ],
      }
  ]
  ```
  正确输出推理内容即可.