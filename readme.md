## Qwen2.5VL本地部署

### 部署流程

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
  # flash-attn==2.6.1
  ```
  注意要注释掉
  ```shell
  git+https://github.com/huggingface/transformers.git
  ```
  **因为huggingface大概率需要挂梯子**
  <br>

- 安装模型本体
  魔搭社区(huggingface也行):https://modelscope.cn/collections/Qwen25-VL-58fbb5d31f1d47
  找到相应的硬件带得动的模型大小.

  这里我用git安装(**缺点：无可视化进度**)，魔搭社区也有其他安装方法.
  ```shell
  git clone https://www.modelscope.cn/Qwen/Qwen2.5-VL-7B-Instruct.git
  ```
  <br>
- 链接模型文件到Qwen2.5Vl本体(option/可选)
  ```shell
  cd Qwen2.5VL
  ln -s <real_model_path> models
  ```
  然后可以在Qwen2.5本体处调用模型了，较为方便.

### demo测试

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