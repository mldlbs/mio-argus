"""
Computer Use Model FastAPI 部署服务
提供模型推理 API 接口
"""
import os
import io
import base64
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

# 导入模型
from model_computer_use_v3 import (
    ComputerUseModelV3,
    ComputerUseModelV3Small,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_image_processor,
    get_tokenizer,
)

# 配置
class Settings(BaseSettings):
    model_path: str = "computer_use_model_v3_real.pth"
    model_type: str = "v3"  # "v3" or "v3_small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_config = ConfigDict(env_file=".env")

settings = Settings()

# 全局变量
model = None
tokenizer = None
image_processor = None
device = torch.device(settings.device)

# 配置日志
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理"""
    global model, tokenizer, image_processor

    logger.info(f"正在加载模型: {settings.model_path}")

    # 加载 tokenizer 和 image processor
    global tokenizer, image_processor
    tokenizer = get_tokenizer()
    image_processor = get_image_processor()

    # 根据配置加载模型
    if settings.model_type == "v3_small":
        model = ComputerUseModelV3Small(num_actions=6, lora_rank=16).to(device)
    else:
        model = ComputerUseModelV3(
            num_actions=6,
            freeze_vit=True,
            freeze_gpt2=True,
        ).to(device)

    # 加载权重
    if os.path.exists(settings.model_path):
        state_dict = torch.load(settings.model_path, map_location=device)
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)
        logger.info(f"模型权重加载成功: {settings.model_path}")
    else:
        logger.warning(f"模型文件不存在: {settings.model_path}，使用随机初始化权重")

    model.eval()
    logger.info(f"模型加载完成，运行在 {device}")

    yield

    logger.info("服务关闭")


app = FastAPI(
    title="Computer Use Model API",
    description="Computer Use Model 推理服务 - 基于 ViT + GPT-2 架构",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求/响应模型
class PredictRequest(BaseModel):
    image_base64: str
    instruction: str
    max_new_tokens: Optional[int] = 50


class PredictResponse(BaseModel):
    action: str
    confidence: float
    coord: Optional[List[float]] = None  # [x, y] 归一化 0-1
    scroll: Optional[str] = None  # "up", "down", "none"
    all_probs: Optional[Dict[str, float]] = None


class BatchPredictRequest(BaseModel):
    images_base64: List[str]
    instructions: List[str]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_type: str


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """预处理图像"""
    img = image.resize((224, 224))
    arr = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    return arr


def decode_base64_image(base64_str: str) -> Image.Image:
    """解码 base64 图片"""
    # 处理可能的 data URL 前缀
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert('RGB')


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        model_type=settings.model_type,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """单样本预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        image = decode_base64_image(request.image_base64)
        image_tensor = preprocess_image(image).unsqueeze(0).to(device)

        encoding = tokenizer(
            request.instruction,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)

        action_logits = outputs['action_logits']
        probs = F.softmax(action_logits, dim=-1)
        pred_id = action_logits.argmax(dim=-1).item()
        confidence = probs[0, pred_id].item()
        action = ACTION_MAP[pred_id]

        coord = outputs['coord_pred'][0].cpu().flatten().tolist()

        scroll_id = outputs['scroll_logits'].argmax(dim=-1).item()
        scroll = ['up', 'down', 'none'][scroll_id]

        all_probs = {ACTION_MAP[i]: float(probs[0, i].item()) for i in range(6)}

        return PredictResponse(
            action=action,
            confidence=float(confidence),
            coord=coord,
            scroll=scroll,
            all_probs=all_probs,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"预测失败: {e}\n{tb}")
        print(f"TRACEBACK: {tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """批量预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if len(request.images_base64) != len(request.instructions):
        raise HTTPException(status_code=400, detail="图片数量与指令数量不匹配")

    results = []
    for img_b64, instr in zip(request.images_base64, request.instructions):
        try:
            image = decode_base64_image(img_b64)
            image_tensor = preprocess_image(image).unsqueeze(0).to(device)

            encoding = tokenizer(
                instr,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(
                    image_tensor, input_ids, attention_mask
                )

                action_logits = outputs['action_logits']
                probs = F.softmax(action_logits, dim=-1)
                pred_id = action_logits.argmax(dim=-1).item()
                confidence = probs[0, pred_id].item()
                action = ACTION_MAP[pred_id]
                coord = outputs['coord_pred'][0].cpu().flatten().tolist()
                scroll_id = outputs['scroll_logits'].argmax(dim=-1).item()
                scroll = ['up', 'down', 'none'][scroll_id]
                all_probs = {ACTION_MAP[i]: float(probs[0, i].item()) for i in range(6)}

                results.append(PredictResponse(
                    action=action,
                    confidence=confidence,
                    coord=coord,
                    scroll=scroll,
                    all_probs=all_probs,
                ))
        except Exception as e:
            logger.error(f"批量预测单个失败: {e}")
            results.append(PredictResponse(
                action="error",
                confidence=0.0,
                coord=None,
                scroll=None,
                all_probs=None,
            ))

    return BatchPredictResponse(predictions=results)


@app.post("/predict/file", response_model=PredictResponse)
async def predict_file(
    image: UploadFile = File(...),
    instruction: str = Form(...),
):
    """文件上传预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        contents = await image.read()
        image_obj = Image.open(io.BytesIO(contents)).convert('RGB')

        image_tensor = preprocess_image(image_obj).unsqueeze(0).to(device)

        encoding = tokenizer(
            instruction,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)
            action_logits = outputs['action_logits']
            probs = F.softmax(action_logits, dim=-1)
            pred_id = action_logits.argmax(dim=-1).item()
            confidence = probs[0, pred_id].item()
            action = ACTION_MAP[pred_id]
            coord = outputs['coord_pred'][0].cpu().flatten().tolist()
            scroll_id = outputs['scroll_logits'].argmax(dim=-1).item()
            scroll = ['up', 'down', 'none'][scroll_id]
            all_probs = {ACTION_MAP[i]: float(probs[0, i].item()) for i in range(6)}

            return PredictResponse(
                action=action,
                confidence=confidence,
                coord=coord,
                scroll=scroll,
                all_probs=all_probs,
            )
    except Exception as e:
        logger.error(f"文件预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/model/info")
async def model_info():
    """模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_type": settings.model_type,
        "num_actions": 6,
        "action_map": ACTION_MAP,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "model_path": settings.model_path,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
